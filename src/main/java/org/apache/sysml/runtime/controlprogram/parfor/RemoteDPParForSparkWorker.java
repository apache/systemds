/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.runtime.controlprogram.parfor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;

import org.apache.hadoop.io.Writable;
import org.apache.spark.Accumulator;
import org.apache.spark.TaskContext;
import org.apache.spark.api.java.function.PairFlatMapFunction;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.parfor.Task.TaskType;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDHandler;
import org.apache.sysml.runtime.controlprogram.parfor.util.PairWritableBlock;
import org.apache.sysml.runtime.controlprogram.parfor.util.PairWritableCell;
import org.apache.sysml.runtime.instructions.cp.IntObject;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.LocalFileUtils;

import scala.Tuple2;

/**
 * 
 */
public class RemoteDPParForSparkWorker extends ParWorker implements PairFlatMapFunction<Iterator<Tuple2<Long, Iterable<Writable>>>, Long, String> 
{
	private static final long serialVersionUID = 30223759283155139L;
	
	private String  _prog = null;
	private boolean _caching = true;
	private String _inputVar = null;
	private String _iterVar = null;
	
	private OutputInfo _oinfo = null;
	private int _rlen = -1;
	private int _clen = -1;
	private int _brlen = -1;
	private int _bclen = -1;
	private boolean _tSparseCol = false;
	private PDataPartitionFormat _dpf = null;
	
	private Accumulator<Integer> _aTasks = null;
	private Accumulator<Integer> _aIters = null;
	
	public RemoteDPParForSparkWorker(String program, String inputVar, String iterVar, boolean cpCaching, MatrixCharacteristics mc, boolean tSparseCol, PDataPartitionFormat dpf, OutputInfo oinfo, Accumulator<Integer> atasks, Accumulator<Integer> aiters) 
		throws DMLRuntimeException
	{
		//keep inputs (unfortunately, spark does not expose task ids and it would be implementation-dependent
		//when this constructor is actually called; hence, we do lazy initialization on task execution)
		_prog = program;
		_caching = cpCaching;
		_inputVar = inputVar;
		_iterVar = iterVar;
		_oinfo = oinfo;
		
		//setup spark accumulators
		_aTasks = atasks;
		_aIters = aiters;
		
		//setup matrixblock partition and meta data
		_rlen = (int)mc.getRows();
		_clen = (int)mc.getCols();
		_brlen = mc.getRowsPerBlock();
		_bclen = mc.getColsPerBlock();
		_tSparseCol = tSparseCol;
		_dpf = dpf;
		switch( _dpf ) { //create matrix partition for reuse
			case ROW_WISE:    _rlen = 1; break;
			case COLUMN_WISE: _clen = 1; break;
			default:  throw new RuntimeException("Partition format not yet supported in fused partition-execute: "+dpf);
		}
	}
	
	@Override 
	public Iterable<Tuple2<Long, String>> call(Iterator<Tuple2<Long, Iterable<Writable>>> arg0)
		throws Exception 
	{
		ArrayList<Tuple2<Long,String>> ret = new ArrayList<Tuple2<Long,String>>();
		
		//lazy parworker initialization
		configureWorker( TaskContext.get().taskAttemptId() ); //requires Spark 1.3
	
		//process all matrix partitions of this data partition
		while( arg0.hasNext() )
		{
			Tuple2<Long,Iterable<Writable>> larg = arg0.next();
			
			//collect input partition (check via equals because oinfo deserialized instance)
			MatrixBlock partition = null;
			if( _oinfo.equals(OutputInfo.BinaryBlockOutputInfo) )
				partition = collectBinaryBlock( larg._2() );
			else
				partition = collectBinaryCellInput( larg._2() );
			
			//update in-memory matrix partition
			MatrixObject mo = (MatrixObject)_ec.getVariable( _inputVar );
			mo.setInMemoryPartition( partition );
					
			//create tasks for input data
			Task lTask = new Task(TaskType.SET);
			lTask.addIteration( new IntObject(_iterVar, larg._1()) );
						
			//execute program
			long numIter = getExecutedIterations();
			super.executeTask( lTask );
					
			//maintain accumulators
			_aTasks.add( 1 );
			_aIters.add( (int)(getExecutedIterations()-numIter) );
			
			//write output if required (matrix indexed write) 
			//note: this copy is necessary for environments without spark libraries
			ArrayList<String> tmp = RemoteParForUtils.exportResultVariables( _workerID, _ec.getVariables(), _resultVars );
			for( String val : tmp )
				ret.add(new Tuple2<Long,String>(_workerID, val));
		}	
		
		return ret;
	}
	
	/**
	 * 
	 * @param ID
	 * @throws DMLRuntimeException 
	 * @throws IOException 
	 */
	private void configureWorker( long ID ) 
		throws DMLRuntimeException, IOException
	{
		_workerID = ID;
		
		//parse and setup parfor body program
		ParForBody body = ProgramConverter.parseParForBody(_prog, (int)_workerID);
		_childBlocks = body.getChildBlocks();
		_ec          = body.getEc();				
		_resultVars  = body.getResultVarNames();
		_numTasks    = 0;
		_numIters    = 0;

		//init local cache manager 
		if( !CacheableData.isCachingActive() ) {
			String uuid = IDHandler.createDistributedUniqueID();
			LocalFileUtils.createWorkingDirectoryWithUUID( uuid );
			CacheableData.initCaching( uuid ); //incl activation, cache dir creation (each map task gets its own dir for simplified cleanup)
		}		
		if( !CacheableData.cacheEvictionLocalFilePrefix.contains("_") ){ //account for local mode
			CacheableData.cacheEvictionLocalFilePrefix = CacheableData.cacheEvictionLocalFilePrefix +"_" + _workerID; 
		}
		
		//ensure that resultvar files are not removed
		super.pinResultVariables();
		
		//enable/disable caching (if required)
		if( !_caching )
			CacheableData.disableCaching();
	}
	
	/**
	 * Collects a matrixblock partition from a given input iterator over 
	 * binary blocks.
	 * 
	 * Note it reuses the instance attribute _partition - multiple calls
	 * will overwrite the result.
	 * 
	 * @param valueList
	 * @return
	 * @throws IOException 
	 */
	private MatrixBlock collectBinaryBlock( Iterable<Writable> valueList ) 
		throws IOException 
	{
		MatrixBlock partition = null;
		
		try
		{
			//reset reuse block, keep configured representation
			if( _tSparseCol )
				partition = new MatrixBlock(_clen, _rlen, true);
			else
				partition = new MatrixBlock(_rlen, _clen, false);

			for( Writable val : valueList )
			{
				PairWritableBlock pairValue = (PairWritableBlock) val;
				int row_offset = (int)(pairValue.indexes.getRowIndex()-1)*_brlen;
				int col_offset = (int)(pairValue.indexes.getColumnIndex()-1)*_bclen;
				MatrixBlock block = pairValue.block;
				if( !partition.isInSparseFormat() ) //DENSE
				{
					partition.copy( row_offset, row_offset+block.getNumRows()-1, 
							   col_offset, col_offset+block.getNumColumns()-1,
							   pairValue.block, false ); 
				}
				else //SPARSE 
				{
					partition.appendToSparse(pairValue.block, row_offset, col_offset);
				}
			}

			//final partition cleanup
			cleanupCollectedMatrixPartition( partition, partition.isInSparseFormat() );
		}
		catch(DMLRuntimeException ex)
		{
			throw new IOException(ex);
		}
		
		return partition;
	}
	
	
	/**
	 * Collects a matrixblock partition from a given input iterator over 
	 * binary cells.
	 * 
	 * Note it reuses the instance attribute _partition - multiple calls
	 * will overwrite the result.
	 * 
	 * @param valueList
	 * @return
	 * @throws IOException 
	 */
	private MatrixBlock collectBinaryCellInput( Iterable<Writable> valueList ) 
		throws IOException 
	{
		MatrixBlock partition = null;

		//reset reuse block, keep configured representation
		if( _tSparseCol )
			partition = new MatrixBlock(_clen, _rlen, true);
		else
			partition = new MatrixBlock(_rlen, _clen, false);
		
		switch( _dpf )
		{
			case ROW_WISE:
				while( valueList.iterator().hasNext() )
				{
					PairWritableCell pairValue = (PairWritableCell)valueList.iterator().next();
					if( pairValue.indexes.getColumnIndex()<0 )
						continue; //cells used to ensure empty partitions
					partition.quickSetValue(0, (int)pairValue.indexes.getColumnIndex()-1, pairValue.cell.getValue());
				}
				break;
			case COLUMN_WISE:
				while( valueList.iterator().hasNext() )
				{
					PairWritableCell pairValue = (PairWritableCell)valueList.iterator().next();
					if( pairValue.indexes.getRowIndex()<0 )
						continue; //cells used to ensure empty partitions
					if( _tSparseCol )
						partition.appendValue(0,(int)pairValue.indexes.getRowIndex()-1, pairValue.cell.getValue());
					else
						partition.quickSetValue((int)pairValue.indexes.getRowIndex()-1, 0, pairValue.cell.getValue());
				}
				break;
			default: 
				throw new IOException("Partition format not yet supported in fused partition-execute: "+_dpf);
		}
		
		//final partition cleanup
		cleanupCollectedMatrixPartition(partition, _tSparseCol);
		
		return partition;
	}
	
	/**
	 * 
	 * @param sort
	 * @throws IOException
	 */
	private void cleanupCollectedMatrixPartition(MatrixBlock partition, boolean sort) 
		throws IOException
	{
		//sort sparse row contents if required
		if( partition.isInSparseFormat() && sort )
			partition.sortSparseRows();

		//ensure right number of nnz
		if( !partition.isInSparseFormat() )
			partition.recomputeNonZeros();
			
		//exam and switch dense/sparse representation
		try {
			partition.examSparsity();
		}
		catch(Exception ex){
			throw new IOException(ex);
		}
	}
}
