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

package org.apache.sysds.runtime.controlprogram.parfor;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

import java.util.ArrayList;

/**
 * Local in-memory realization of result merge. If the resulting matrix is
 * small enough to fit into the JVM memory, this class can be used for efficient 
 * serial or multi-threaded merge.
 * 
 * 
 */
public class ResultMergeLocalMemory extends ResultMergeMatrix
{
	private static final long serialVersionUID = -3543612508601511701L;
	
	//internal comparison matrix
	private DenseBlock _compare = null;
	
	public ResultMergeLocalMemory( MatrixObject out, MatrixObject[] in, String outputFilename, boolean accum ) {
		super( out, in, outputFilename, accum );
	}
	
	@Override
	public MatrixObject executeSerialMerge() 
	{
		MatrixObject moNew = null; //always create new matrix object (required for nested parallelism)
		
		if( LOG.isTraceEnabled() )
			LOG.trace("ResultMerge (local, in-memory): Execute serial merge for output "
				+_output.hashCode()+" (fname="+_output.getFileName()+")");
		
		try
		{
			//get old output matrix from cache for compare
			MatrixBlock outMB = _output.acquireRead();
			
			//create output matrices in correct format according to 
			//the estimated number of non-zeros
			long estnnz = getOutputNnzEstimate();
			MatrixBlock outMBNew = new MatrixBlock(outMB.getNumRows(), 
				outMB.getNumColumns(), estnnz).allocateBlock();
			boolean appendOnly = outMBNew.isInSparseFormat();
			
			//create compare matrix if required (existing data in result)
			_compare = getCompareMatrix(outMB);
			if( _compare != null || _isAccum )
				outMBNew.copy(outMB);
			
			//serial merge all inputs
			boolean flagMerged = false;
			for( MatrixObject in : _inputs )
			{
				//check for empty inputs (no iterations executed)
				if( in != null && in != _output ) 
				{
					if( LOG.isTraceEnabled() )
						LOG.trace("ResultMerge (local, in-memory): Merge input "+in.hashCode()+" (fname="+in.getFileName()+")");
					
					//read/pin input_i
					MatrixBlock inMB = in.acquireRead();
					
					//core merge 
					merge( outMBNew, inMB, _compare, appendOnly );
					
					//unpin and clear in-memory input_i
					in.release();
					in.clearData();
					flagMerged = true;
					
					//determine need for sparse2dense change during merge
					boolean sparseToDense = appendOnly && !MatrixBlock.evalSparseFormatInMemory(
						outMBNew.getNumRows(), outMBNew.getNumColumns(), outMBNew.getNonZeros()); 
					if( sparseToDense ) {
						outMBNew.sortSparseRows(); //sort sparse due to append-only
						outMBNew.examSparsity(); //sparse-dense representation change
						appendOnly = false; //change merge state for subsequent inputs
					}
				}
			}
		
			//sort sparse due to append-only
			if( appendOnly && !_isAccum )
				outMBNew.sortSparseRows();
			
			//change sparsity if required after 
			outMBNew.examSparsity(); 
			
			//create output
			if( flagMerged ) {
				//create new output matrix 
				//(e.g., to prevent potential export<->read file access conflict in specific cases of 
				// local-remote nested parfor))
				moNew = createNewMatrixObject( outMBNew );
			}
			else {
				moNew = _output; //return old matrix, to prevent copy
			}
			
			//release old output, and all inputs
			_output.release();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}

		//LOG.trace("ResultMerge (local, in-memory): Executed serial merge for output "+_output.getVarName()+" (fname="+_output.getFileName()+") in "+time.stop()+"ms");
		
		return moNew;
	}
	
	@Override
	public MatrixObject executeParallelMerge( int par ) 
	{
		MatrixObject moNew = null; //always create new matrix object (required for nested parallelism)
		
		if( LOG.isTraceEnabled() )
			LOG.trace("ResultMerge (local, in-memory): Execute parallel (par="+par+") "
				+ "merge for output "+_output.hashCode()+" (fname="+_output.getFileName()+")");
		
		try
		{
			//get matrix blocks through caching 
			MatrixBlock outMB = _output.acquireRead();
			ArrayList<MatrixObject> inMO = new ArrayList<>();
			for( MatrixObject in : _inputs ) {
				//check for empty inputs (no iterations executed)
				if( in !=null && in != _output ) 
					inMO.add( in );
			}
			
			if( !inMO.isEmpty() ) //if there exist something to merge
			{
				//get old output matrix from cache for compare
				//NOTE: always in dense representation in order to allow for parallel unsynchronized access 
				long rows = outMB.getNumRows();
				long cols = outMB.getNumColumns();
				MatrixBlock outMBNew = new MatrixBlock((int)rows, (int)cols, false);
				outMBNew.allocateDenseBlockUnsafe((int)rows, (int)cols);
				
				//create compare matrix if required (existing data in result)
				_compare = getCompareMatrix(outMB);
				if( _compare != null || _isAccum )
					outMBNew.copy(outMB);
				
				//parallel merge of all inputs

				int numThreads = Math.min(par, inMO.size()); //number of inputs can be lower than par
				numThreads = Math.min(numThreads, InfrastructureAnalyzer.getLocalParallelism()); //ensure robustness for remote exec
				Thread[] threads = new Thread[ numThreads ];
				for( int k=0; k<inMO.size(); k+=numThreads ) //multiple waves if necessary
				{
					//create and start threads
					for( int i=0; i<threads.length; i++ )
					{
						ResultMergeWorker rmw = new ResultMergeWorker(inMO.get(k+i), outMBNew);
						threads[i] = new Thread(rmw);
						threads[i].setPriority(Thread.MAX_PRIORITY);
						threads[i].start(); // start execution
					}	
					//wait for all workers to finish
					for( int i=0; i<threads.length; i++ )
					{
						threads[i].join();
					}
				}
				
				//create new output matrix 
				//(e.g., to prevent potential export<->read file access conflict in specific cases of 
				// local-remote nested parfor))
				moNew = createNewMatrixObject( outMBNew );
			}
			else {
				moNew = _output; //return old matrix, to prevent copy
			}
			
			//release old output, and all inputs
			_output.release();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		
		//LOG.trace("ResultMerge (local, in-memory): Executed parallel (par="+par+") merge for output "+_output.getVarName()+" (fname="+_output.getFileName()+") in "+time.stop()+"ms");

		return moNew;
	}

	private DenseBlock getCompareMatrix( MatrixBlock output ) {
		//create compare matrix only if required
		if( !output.isEmptyBlock(false) )
			return DataConverter.convertToDenseBlock(output, false);
		return null;
	}

	private MatrixObject createNewMatrixObject( MatrixBlock data ) {
		ValueType vt = _output.getValueType();
		MetaDataFormat metadata = (MetaDataFormat) _output.getMetaData();
		MatrixObject moNew = new MatrixObject( vt, _outputFName );
		
		//create deep copy of metadata obj
		DataCharacteristics mcOld = metadata.getDataCharacteristics();
		MatrixCharacteristics mc = new MatrixCharacteristics(mcOld);
		mc.setNonZeros(data.getNonZeros());
		moNew.setMetaData(new MetaDataFormat(mc, metadata.getFileFormat()));
		
		//adjust dense/sparse representation
		data.examSparsity();
		
		//release new output
		moNew.acquireModify(data);
		moNew.release();
		
		return moNew;
	}

	
	/**
	 * Merges <code>in</code> into <code>out</code> by inserting all non-zeros of <code>in</code>
	 * into <code>out</code> at their given positions. This is an update-in-place.
	 * 
	 * NOTE: similar to converters, but not directly applicable as we are interested in combining
	 * two objects with each other; not unary transformation.
	 * 
	 * @param out output matrix block
	 * @param in input matrix block
	 * @param compare initialized output
	 * @param appendOnly ?
	 */
	private void merge( MatrixBlock out, MatrixBlock in, DenseBlock compare, boolean appendOnly ) {
		if( _compare == null || _isAccum )
			mergeWithoutComp(out, in, _compare, appendOnly, true);
		else
			mergeWithComp(out, in, _compare);
	}
	
	/**
	 * Estimates the number of non-zeros in the final merged output.
	 * For scenarios without compare matrix, this is the exact number 
	 * of non-zeros due to guaranteed disjoint results per worker.
	 * 
	 * @return estimated number of non-zeros.
	 */
	private long getOutputNnzEstimate() {
		long nnzInputs = 0;
		for( MatrixObject input : _inputs )
			if( input != null )
				nnzInputs += Math.max(input.getNnz(),1);
		long rlen = _output.getNumRows();
		long clen = _output.getNumColumns();
		return Math.min(rlen * clen,
			Math.max(nnzInputs, _output.getNnz()));
	}
	
	
	/**
	 * NOTE: only used if matrix in dense
	 */
	private class ResultMergeWorker implements Runnable
	{
		private MatrixObject _inMO  = null;
		private MatrixBlock  _outMB = null;
		
		public ResultMergeWorker(MatrixObject inMO, MatrixBlock outMB)
		{
			_inMO  = inMO;
			_outMB = outMB;
		}

		@Override
		public void run() 
		{
			//read each input if required
			try
			{
				LOG.trace("ResultMerge (local, in-memory): Merge input "+_inMO.hashCode()+" (fname="+_inMO.getFileName()+")");
				
				MatrixBlock inMB = _inMO.acquireRead(); //incl. implicit read from HDFS
				merge( _outMB, inMB, _compare, false );
				_inMO.release();
				_inMO.clearData();
			}
			catch(Exception ex)
			{
				throw new RuntimeException("Failed to parallel merge result.", ex);
			}
		}
		
	}
}
