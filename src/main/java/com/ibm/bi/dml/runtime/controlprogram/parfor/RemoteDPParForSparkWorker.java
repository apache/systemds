/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.hadoop.io.Writable;
import org.apache.spark.Accumulator;
import org.apache.spark.TaskContext;
import org.apache.spark.api.java.function.PairFlatMapFunction;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheableData;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.Task.TaskType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDHandler;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.PairWritableBlock;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.PairWritableCell;
import com.ibm.bi.dml.runtime.instructions.cp.IntObject;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;

import scala.Tuple2;

/**
 * 
 * 
 */
public class RemoteDPParForSparkWorker extends ParWorker implements PairFlatMapFunction<Tuple2<Long, Iterable<Writable>>, Long, String> 
{

	private static final long serialVersionUID = 30223759283155139L;
	
	private boolean _initialized = false;
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
	
	private MatrixBlock _partition = null;
	
	public RemoteDPParForSparkWorker(String program, String inputVar, String iterVar, boolean cpCaching, int rlen, int clen, int brlen, int bclen, boolean tSparseCol, PDataPartitionFormat dpf, OutputInfo oinfo, Accumulator<Integer> atasks, Accumulator<Integer> aiters) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		//keep inputs (unfortunately, spark does not expose task ids and it would be implementation-dependent
		//when this constructor is actually called; hence, we do lazy initialization on task execution)
		_initialized = false;
		_prog = program;
		_caching = cpCaching;
		_inputVar = inputVar;
		_iterVar = iterVar;
		_oinfo = oinfo;
		
		//setup spark accumulators
		_aTasks = atasks;
		_aIters = aiters;
		
		//setup matrixblock partition and meta data
		_rlen = rlen;
		_clen = clen;
		_brlen = brlen;
		_bclen = bclen;
		_tSparseCol = tSparseCol;
		_dpf = dpf;
		switch( _dpf ) { //create matrix partition for reuse
			case ROW_WISE:    _rlen = 1; break;
			case COLUMN_WISE: _clen = 1; break;
			default:  throw new RuntimeException("Partition format not yet supported in fused partition-execute: "+dpf);
		}
		if( _tSparseCol )
			_partition = new MatrixBlock(_clen, _rlen, true);
		else
			_partition = new MatrixBlock(_rlen, _clen, false);
	}
	
	@Override 
	public Iterable<Tuple2<Long, String>> call(Tuple2<Long, Iterable<Writable>> arg0)
		throws Exception 
	{
		//lazy parworker initialization
		if( !_initialized )
			configureWorker( TaskContext.get().taskAttemptId() ); //requires Spark 1.3
	
		//collect input partition (check via equals because oinfo deserialized instance)
		if( _oinfo.equals(OutputInfo.BinaryBlockOutputInfo) )
			_partition = collectBinaryBlock( arg0._2() );
		else
			_partition = collectBinaryCellInput( arg0._2() );
				
		
		//update in-memory matrix partition
		MatrixObject mo = (MatrixObject)_ec.getVariable( _inputVar );
		mo.setInMemoryPartition( _partition );
				
		//create tasks for input data
		Task lTask = new Task(TaskType.SET);
		lTask.addIteration( new IntObject(_iterVar, arg0._1()) );
					
		//execute program
		long numIter = getExecutedIterations();
		super.executeTask( lTask );
				
		//maintain accumulators
		_aTasks.add( 1 );
		_aIters.add( (int)(getExecutedIterations()-numIter) );
		
		//write output if required (matrix indexed write) 
		//note: this copy is necessary for environments without spark libraries
		ArrayList<Tuple2<Long,String>> ret = new ArrayList<Tuple2<Long,String>>();
		ArrayList<String> tmp = RemoteParForUtils.exportResultVariables( _workerID, _ec.getVariables(), _resultVars );
		for( String val : tmp )
			ret.add(new Tuple2<Long,String>(_workerID, val));
			
		return ret;
	}
	
	/**
	 * 
	 * @param ID
	 * @throws DMLUnsupportedOperationException 
	 * @throws DMLRuntimeException 
	 * @throws IOException 
	 */
	private void configureWorker( long ID ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException, IOException
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
		if( !CacheableData.isCachingActive() ) 
		{
			String uuid = IDHandler.createDistributedUniqueID();
			LocalFileUtils.createWorkingDirectoryWithUUID( uuid );
			CacheableData.initCaching( uuid ); //incl activation, cache dir creation (each map task gets its own dir for simplified cleanup)
		}		
		if( !CacheableData.cacheEvictionLocalFilePrefix.contains("_") ) //account for local mode
		{
			CacheableData.cacheEvictionLocalFilePrefix = CacheableData.cacheEvictionLocalFilePrefix +"_" + _workerID; 
			CacheableData.cacheEvictionHDFSFilePrefix = CacheableData.cacheEvictionHDFSFilePrefix +"_" + _workerID;
		}
		
		//ensure that resultvar files are not removed
		super.pinResultVariables();
		
		//enable/disable caching (if required)
		if( !_caching )
			CacheableData.disableCaching();
		
		//make as lazily intialized
		_initialized = true;
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
		try
		{
			//reset reuse block, keep configured representation
			_partition.reset((int)_rlen, (int)_clen);	

			for( Writable val : valueList )
			{
				PairWritableBlock pairValue = (PairWritableBlock) val;
				int row_offset = (int)(pairValue.indexes.getRowIndex()-1)*_brlen;
				int col_offset = (int)(pairValue.indexes.getColumnIndex()-1)*_bclen;
				MatrixBlock block = pairValue.block;
				if( !_partition.isInSparseFormat() ) //DENSE
				{
					_partition.copy( row_offset, row_offset+block.getNumRows()-1, 
							   col_offset, col_offset+block.getNumColumns()-1,
							   pairValue.block, false ); 
				}
				else //SPARSE 
				{
					_partition.appendToSparse(pairValue.block, row_offset, col_offset);
				}
			}

			//final partition cleanup
			cleanupCollectedMatrixPartition( _partition.isInSparseFormat() );
		}
		catch(DMLRuntimeException ex)
		{
			throw new IOException(ex);
		}
		
		return _partition;
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
		//reset reuse block, keep configured representation
		if( _tSparseCol )
			_partition.reset(_clen, _rlen);	
		else
			_partition.reset(_rlen, _clen);
		
		switch( _dpf )
		{
			case ROW_WISE:
				while( valueList.iterator().hasNext() )
				{
					PairWritableCell pairValue = (PairWritableCell)valueList.iterator().next();
					if( pairValue.indexes.getColumnIndex()<0 )
						continue; //cells used to ensure empty partitions
					_partition.quickSetValue(0, (int)pairValue.indexes.getColumnIndex()-1, pairValue.cell.getValue());
				}
				break;
			case COLUMN_WISE:
				while( valueList.iterator().hasNext() )
				{
					PairWritableCell pairValue = (PairWritableCell)valueList.iterator().next();
					if( pairValue.indexes.getRowIndex()<0 )
						continue; //cells used to ensure empty partitions
					if( _tSparseCol )
						_partition.appendValue(0,(int)pairValue.indexes.getRowIndex()-1, pairValue.cell.getValue());
					else
						_partition.quickSetValue((int)pairValue.indexes.getRowIndex()-1, 0, pairValue.cell.getValue());
				}
				break;
			default: 
				throw new IOException("Partition format not yet supported in fused partition-execute: "+_dpf);
		}
		
		//final partition cleanup
		cleanupCollectedMatrixPartition(_tSparseCol);
		
		return _partition;
	}
	
	/**
	 * 
	 * @param sort
	 * @throws IOException
	 */
	private void cleanupCollectedMatrixPartition(boolean sort) 
		throws IOException
	{
		//sort sparse row contents if required
		if( _partition.isInSparseFormat() && sort )
			_partition.sortSparseRows();

		//ensure right number of nnz
		if( !_partition.isInSparseFormat() )
			_partition.recomputeNonZeros();
			
		//exam and switch dense/sparse representation
		try {
			_partition.examSparsity();
		}
		catch(Exception ex){
			throw new IOException(ex);
		}
	}
}
