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

package org.apache.sysml.runtime.controlprogram.parfor;

import java.util.ArrayList;

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.DataConverter;

/**
 * Local in-memory realization of result merge. If the resulting matrix is
 * small enough to fit into the JVM memory, this class can be used for efficient 
 * serial or multi-threaded merge.
 * 
 * 
 */
public class ResultMergeLocalMemory extends ResultMerge
{	
	
	//internal comparison matrix
	private double[][]        _compare     = null;
	
	public ResultMergeLocalMemory( MatrixObject out, MatrixObject[] in, String outputFilename )
	{
		super( out, in, outputFilename );
	}
	
	@Override
	public MatrixObject executeSerialMerge() 
		throws DMLRuntimeException
	{
		MatrixObject moNew = null; //always create new matrix object (required for nested parallelism)

		LOG.trace("ResultMerge (local, in-memory): Execute serial merge for output "+_output.getVarName()+" (fname="+_output.getFileName()+")");
				
		try
		{
			//get matrix blocks through caching 
			MatrixBlock outMB = _output.acquireRead();
			
			//get old output matrix from cache for compare
			int estnnz = outMB.getNumRows()*outMB.getNumColumns();
			MatrixBlock outMBNew = new MatrixBlock(outMB.getNumRows(), outMB.getNumColumns(), 
					                               outMB.isInSparseFormat(), estnnz);
			boolean appendOnly = outMBNew.isInSparseFormat();
			
			//create compare matrix if required (existing data in result)
			_compare = createCompareMatrix(outMB);
			if( _compare != null )
				outMBNew.copy(outMB);
			
			//serial merge all inputs
			boolean flagMerged = false;
			for( MatrixObject in : _inputs )
			{
				//check for empty inputs (no iterations executed)
				if( in !=null && in != _output ) 
				{
					LOG.trace("ResultMerge (local, in-memory): Merge input "+in.getVarName()+" (fname="+in.getFileName()+")");
					
					//read/pin input_i
					MatrixBlock inMB = in.acquireRead();	
					
					//core merge 
					merge( outMBNew, inMB, appendOnly );
					
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
			if( appendOnly )
				outMBNew.sortSparseRows();
			
			//change sparsity if required after 
			outMBNew.examSparsity(); 
			
			//create output
			if( flagMerged )
			{		
				//create new output matrix 
				//(e.g., to prevent potential export<->read file access conflict in specific cases of 
				// local-remote nested parfor))
				moNew = createNewMatrixObject( outMBNew );	
			}
			else
			{
				moNew = _output; //return old matrix, to prevent copy
			}
			
			//release old output, and all inputs
			_output.release();
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}

		//LOG.trace("ResultMerge (local, in-memory): Executed serial merge for output "+_output.getVarName()+" (fname="+_output.getFileName()+") in "+time.stop()+"ms");
		
		return moNew;
	}
	
	@Override
	public MatrixObject executeParallelMerge( int par ) 
		throws DMLRuntimeException
	{		
		MatrixObject moNew = null; //always create new matrix object (required for nested parallelism)
	
		//Timing time = null;
		LOG.trace("ResultMerge (local, in-memory): Execute parallel (par="+par+") merge for output "+_output.getVarName()+" (fname="+_output.getFileName()+")");
		//	time = new Timing();
		//	time.start();
		

		try
		{
			//get matrix blocks through caching 
			MatrixBlock outMB = _output.acquireRead();
			ArrayList<MatrixObject> inMO = new ArrayList<MatrixObject>();
			for( MatrixObject in : _inputs )
			{
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
				_compare = createCompareMatrix(outMB);
				if( _compare != null )
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
			else
			{
				moNew = _output; //return old matrix, to prevent copy
			}
			
			//release old output, and all inputs
			_output.release();			
			//_output.clearData(); //save, since it respects pin/unpin  
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		
		//LOG.trace("ResultMerge (local, in-memory): Executed parallel (par="+par+") merge for output "+_output.getVarName()+" (fname="+_output.getFileName()+") in "+time.stop()+"ms");

		return moNew;		
	}

	/**
	 * 
	 * @param output
	 * @return
	 */
	private double[][] createCompareMatrix( MatrixBlock output )
	{
		double[][] ret = null;
		
		//create compare matrix only if required
		if( output.getNonZeros() > 0 )
		{
			ret = DataConverter.convertToDoubleMatrix( output );
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param varName
	 * @param vt
	 * @param metadata
	 * @param data
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private MatrixObject createNewMatrixObject( MatrixBlock data ) 
		throws DMLRuntimeException
	{
		String varName = _output.getVarName();
		ValueType vt = _output.getValueType();
		MatrixFormatMetaData metadata = (MatrixFormatMetaData) _output.getMetaData();
		
		MatrixObject moNew = new MatrixObject( vt, _outputFName );
		moNew.setVarName( varName.contains(NAME_SUFFIX) ? varName : varName+NAME_SUFFIX );
		moNew.setDataType( DataType.MATRIX );
		
		//create deep copy of metadata obj
		MatrixCharacteristics mcOld = metadata.getMatrixCharacteristics();
		OutputInfo oiOld = metadata.getOutputInfo();
		InputInfo iiOld = metadata.getInputInfo();
		MatrixCharacteristics mc = new MatrixCharacteristics(mcOld.getRows(),mcOld.getCols(),
				                                             mcOld.getRowsPerBlock(),mcOld.getColsPerBlock());
		mc.setNonZeros(data.getNonZeros());
		MatrixFormatMetaData meta = new MatrixFormatMetaData(mc,oiOld,iiOld);
		moNew.setMetaData( meta );
		
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
	 * @param out
	 * @param in
	 * @throws DMLRuntimeException 
	 */
	private void merge( MatrixBlock out, MatrixBlock in, boolean appendOnly ) 
		throws DMLRuntimeException
	{
		if( _compare == null )
			mergeWithoutComp(out, in, appendOnly);
		else
			mergeWithComp(out, in, _compare);
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
				LOG.trace("ResultMerge (local, in-memory): Merge input "+_inMO.getVarName()+" (fname="+_inMO.getFileName()+")");
				
				MatrixBlock inMB = _inMO.acquireRead(); //incl. implicit read from HDFS
				merge( _outMB, inMB, false );
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
