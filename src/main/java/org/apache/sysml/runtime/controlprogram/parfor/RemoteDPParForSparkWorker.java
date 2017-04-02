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
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;

import org.apache.hadoop.io.Writable;
import org.apache.spark.TaskContext;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.util.LongAccumulator;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.codegen.CodegenUtils;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PartitionFormat;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.parfor.Task.TaskType;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDHandler;
import org.apache.sysml.runtime.controlprogram.parfor.util.PairWritableBlock;
import org.apache.sysml.runtime.controlprogram.parfor.util.PairWritableCell;
import org.apache.sysml.runtime.instructions.cp.IntObject;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.LocalFileUtils;

import scala.Tuple2;

public class RemoteDPParForSparkWorker extends ParWorker implements PairFlatMapFunction<Iterator<Tuple2<Long, Iterable<Writable>>>, Long, String> 
{
	private static final long serialVersionUID = 30223759283155139L;
	
	private final String  _prog;
	private final HashMap<String, byte[]> _clsMap;
	private final boolean _caching;
	private final String _inputVar;
	private final String _iterVar;
	
	private final OutputInfo _oinfo;
	private final int _rlen;
	private final int _clen;
	private final int _brlen;
	private final int _bclen;
	private final boolean _tSparseCol;
	private final PDataPartitionFormat _dpf;
	
	private final LongAccumulator _aTasks;
	private final LongAccumulator _aIters;
	
	public RemoteDPParForSparkWorker(String program, HashMap<String, byte[]> clsMap, String inputVar, String iterVar, 
			boolean cpCaching, MatrixCharacteristics mc, boolean tSparseCol, PartitionFormat dpf, OutputInfo oinfo, 
			LongAccumulator atasks, LongAccumulator aiters) 
		throws DMLRuntimeException
	{
		_prog = program;
		_clsMap = clsMap;
		_caching = cpCaching;
		_inputVar = inputVar;
		_iterVar = iterVar;
		_oinfo = oinfo;
		
		//setup spark accumulators
		_aTasks = atasks;
		_aIters = aiters;
		
		//setup matrix block partition meta data
		switch( dpf._dpf ) {
			case ROW_WISE: 
				_rlen = (int)mc.getRows(); _clen = 1; break;
			case ROW_BLOCK_WISE_N:
				_rlen = dpf._N; _clen = (int)mc.getCols(); break;
			case COLUMN_BLOCK_WISE:
				_rlen = 1; _clen = (int)mc.getCols(); break;
			case COLUMN_BLOCK_WISE_N:
				_rlen = (int)mc.getRows(); _clen = dpf._N; break;
			default:
				throw new RuntimeException("Unsupported partition format: "+dpf._dpf.name());
		}
		_brlen = mc.getRowsPerBlock();
		_bclen = mc.getColsPerBlock();
		_tSparseCol = tSparseCol;
		_dpf = dpf._dpf;
	}
	
	@Override 
	public Iterator<Tuple2<Long, String>> call(Iterator<Tuple2<Long, Iterable<Writable>>> arg0)
		throws Exception 
	{
		ArrayList<Tuple2<Long,String>> ret = new ArrayList<Tuple2<Long,String>>();
		
		//lazy parworker initialization
		configureWorker( TaskContext.get().taskAttemptId() );
	
		//process all matrix partitions of this data partition
		MatrixBlock partition = null;
		while( arg0.hasNext() )
		{
			Tuple2<Long,Iterable<Writable>> larg = arg0.next();
			
			//collect input partition (check via equals because oinfo deserialized instance)
			if( _oinfo.equals(OutputInfo.BinaryBlockOutputInfo) )
				partition = collectBinaryBlock( larg._2(), partition );
			else
				partition = collectBinaryCellInput( larg._2() );
			
			//update in-memory matrix partition
			MatrixObject mo = _ec.getMatrixObject( _inputVar );
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
		}
		
		//write output if required (matrix indexed write) 
		ArrayList<String> tmp = RemoteParForUtils.exportResultVariables( _workerID, _ec.getVariables(), _resultVars );
		for( String val : tmp )
			ret.add(new Tuple2<Long,String>(_workerID, val));
		
		return ret.iterator();
	}

	private void configureWorker( long ID ) 
		throws DMLRuntimeException, IOException
	{
		_workerID = ID;
		
		//initialize codegen class cache (before program parsing)
		synchronized( CodegenUtils.class ) {
			for( Entry<String, byte[]> e : _clsMap.entrySet() )
				CodegenUtils.getClass(e.getKey(), e.getValue());
		}
		
		//parse and setup parfor body program
		ParForBody body = ProgramConverter.parseParForBody(_prog, (int)_workerID);
		_childBlocks = body.getChildBlocks();
		_ec          = body.getEc();				
		_resultVars  = body.getResultVarNames();
		_numTasks    = 0;
		_numIters    = 0;

		//init and register-cleanup of buffer pool (in parfor spark, multiple tasks might 
		//share the process-local, i.e., per executor, buffer pool; hence we synchronize 
		//the initialization and immediately register the created directory for cleanup
		//on process exit, i.e., executor exit, including any files created in the future.
		synchronized( CacheableData.class ) {
			if( !CacheableData.isCachingActive() && !InfrastructureAnalyzer.isLocalMode() ) { 
				//create id, executor working dir, and cache dir
				String uuid = IDHandler.createDistributedUniqueID();
				LocalFileUtils.createWorkingDirectoryWithUUID( uuid );
				CacheableData.initCaching( uuid ); //incl activation and cache dir creation
				CacheableData.cacheEvictionLocalFilePrefix = 
						CacheableData.cacheEvictionLocalFilePrefix +"_" + _workerID; 
				//register entire working dir for delete on shutdown
				RemoteParForUtils.cleanupWorkingDirectoriesOnShutdown();
			}	
		}
		
		//ensure that resultvar files are not removed
		super.pinResultVariables();
		
		//enable/disable caching (if required and not in CP process)
		if( !_caching && !InfrastructureAnalyzer.isLocalMode() )
			CacheableData.disableCaching();
	}
	
	/**
	 * Collects a matrixblock partition from a given input iterator over 
	 * binary blocks.
	 * 
	 * Note it reuses the instance attribute _partition - multiple calls
	 * will overwrite the result.
	 * 
	 * @param valueList iterable writables
	 * @param reuse matrix block partition for reuse
	 * @return matrix block
	 * @throws IOException if IOException occurs
	 */
	private MatrixBlock collectBinaryBlock( Iterable<Writable> valueList, MatrixBlock reuse ) 
		throws IOException 
	{
		//fast path for partition of single fragment (see pseudo grouping),
		//which avoids unnecessary copies and reduces memory pressure
		if( valueList instanceof Collection && ((Collection<Writable>)valueList).size()==1 ) {
			return ((PairWritableBlock)valueList.iterator().next()).block;
		}
		
		//default: create or reuse target partition and copy individual partition fragments 
		//into this target, including nnz maintenance and potential dense-sparse format change 
		MatrixBlock partition = reuse;
		
		try
		{
			//reset reuse block, keep configured representation
			if( _tSparseCol )
				partition = new MatrixBlock(_clen, _rlen, true);
			else if( partition!=null )
				partition.reset(_rlen, _clen, false);
			else
				partition = new MatrixBlock(_rlen, _clen, false);

			long lnnz = 0;
			for( Writable val : valueList ) {
				PairWritableBlock pval = (PairWritableBlock) val;
				int row_offset = (int)(pval.indexes.getRowIndex()-1)*_brlen;
				int col_offset = (int)(pval.indexes.getColumnIndex()-1)*_bclen;
				if( !partition.isInSparseFormat() ) //DENSE
					partition.copy( row_offset, row_offset+pval.block.getNumRows()-1, 
							   col_offset, col_offset+pval.block.getNumColumns()-1,
							   pval.block, false ); 
				else //SPARSE 
					partition.appendToSparse(pval.block, row_offset, col_offset);
				lnnz += pval.block.getNonZeros();
			}

			//post-processing: cleanups if required
			if( partition.isInSparseFormat() && _clen>_bclen )
				partition.sortSparseRows();
			partition.setNonZeros(lnnz);
			partition.examSparsity();
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
	 * @param valueList iterable writables
	 * @return matrix block
	 * @throws IOException if IOException occurs
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
		
		//post-processing: cleanups if required
		try {
			if( partition.isInSparseFormat() && _tSparseCol )
				partition.sortSparseRows();
			partition.recomputeNonZeros();
			partition.examSparsity();
		}
		catch(DMLRuntimeException ex) {
			throw new IOException(ex);
		}
			
		return partition;
	}
}
