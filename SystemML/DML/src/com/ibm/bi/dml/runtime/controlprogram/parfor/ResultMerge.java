package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.utils.DMLRuntimeException;

/**
 * Due to independence of all iterations, any result has the following properties:
 * (1) non local var, (2) matrix object, and (3) completely independent.
 * These properties allow us to realize result merging in parallel without any synchronization. 
 * 
 * RESTRICTIONS: 
 *   * assumption that each result matrix fits entirely in main-memory
 *   * currently no reduction aggregations functions
 * 
 */
public class ResultMerge 
{
	private MatrixObjectNew   _output = null;
	private MatrixObjectNew[] _inputs = null; 
	
	private double[][]        _compare = null;
	
	public ResultMerge( MatrixObjectNew out, MatrixObjectNew[] in )
	{
		_output = out;
		_inputs = in;
	}
	
	/**
	 * Merge all given input matrices sequentially into the given output matrix.
	 * The required space in-memory is the size of the output matrix plus the size
	 * of one input matrix at a time.
	 * 
	 * @return output (merged) matrix
	 * @throws DMLRuntimeException
	 */
	public MatrixObjectNew executeSerialMerge() 
		throws DMLRuntimeException
	{
		try
		{
			//get output matrix from cache 
			MatrixBlock outMB = _output.acquireModify();
	
			//create compare matrix if required
			_compare = createCompareMatrix(outMB);
						
			for( MatrixObjectNew in : _inputs ) 
			{
				//check for empty inputs (no iterations executed)
				if( in != _output ) 
				{
					//get input matrix from cache
					MatrixBlock inMB = in.acquireRead();
					
					//merge each input to output
					merge( outMB, inMB );
					
					//release input
					in.release();
				}
			}
			
			//release output
			_output.release();
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		
		return _output;
	}
	
	/**
	 * Merge all given input matrices in parallel into the given output matrix.
	 * The required space in-memory is the size of the output matrix plus the size
	 * of all input matrices.
	 * 
	 * @param par degree of parallelism
	 * @return output (merged) matrix
	 * @throws DMLRuntimeException
	 */
	public MatrixObjectNew executeParallelMerge(int par) 
		throws DMLRuntimeException
	{
		try
		{
			//get matrix blocks through caching 
			MatrixBlock outMB = _output.acquireModify();
			ArrayList<MatrixBlock> inMB = new ArrayList<MatrixBlock>();
			for( MatrixObjectNew in : _inputs )
			{
				//check for empty inputs (no iterations executed)
				if( in != _output ) 
					inMB.add( in.acquireRead() );	
			}
			
			//create compare matrix if required
			_compare = createCompareMatrix(outMB);
			
			//create and start threads
			Thread[] threads = new Thread[ inMB.size() ];
			for( int i=0; i<threads.length; i++ )
			{
				ResultMergeWorker rmw = new ResultMergeWorker(inMB.get(i), outMB);
				threads[i] = new Thread(rmw);
				threads[i].setPriority(Thread.MAX_PRIORITY);
				threads[i].start(); // start execution
			}
				
			//wait for all workers to finish
			for( int i=0; i<threads.length; i++ )
			{
				threads[i].join();
			}
			
			//release all data
			_output.release();
			for( MatrixObjectNew in : _inputs )
			{
				//check for empty results
				if( in != _output ) 
					in.release(); //only if required (see above)
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		
		return _output;
	}
	
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
	 * Merges <code>in</code> into <code>out</code> by inserting all non-zeros of <code>in</code>
	 * into <code>out</code> at their given positions. This is an update-in-place.
	 * 
	 * NOTE: similar to converters, but not directly applicable as we are interested in combining
	 * two objects with each other; not unary transformation.
	 * 
	 * @param out
	 * @param in
	 */
	private void merge( MatrixBlock out, MatrixBlock in )
	{
		if( _compare == null )
			mergeWithoutComp(out, in);
		else
			mergeWithComp(out, in);
	}
	
	//TODO fine-gained synchronization in setValue (for sparse and dense) due to dynamic allocations, 
	
	private void mergeWithComp( MatrixBlock out, MatrixBlock in )
	{
		if( in.isInSparseFormat() ) //sparse input format
		{
			HashMap<CellIndex, Double> sparseMap = in.getSparseMap();
			for(Entry<CellIndex,Double> cell : sparseMap.entrySet() )
			{
				CellIndex key = cell.getKey();
				
				double value  = cell.getValue();  //input value
				if(   value != _compare[key.row][key.column]   //for new values only (div)
				   && value != 0     )                         //for all nnz
				{
					out.setValue(key, value);
				}
			}
		}
		else //dense input format
		{
			//for a merge this case will seldom happen, as each input MatrixObject
			//has at most 1/numThreads of all values in it.
			double[] values = in.getDenseArray();
			int rows = in.getNumRows();
			int cols = in.getNumColumns();
			
			for( int i=0; i<rows; i++ )
				for( int j=0; j<cols; j++ )
				{
				    double value = values[i*cols + j];  //input value
					if(   value != _compare[i][j]   //for new values only (div)
					   && value != 0              ) //for all nnz
					{
						out.setValue( i, j, value );	
					}
				}
		}			
	}
	
	private void mergeWithoutComp( MatrixBlock out, MatrixBlock in )
	{
		if( in.isInSparseFormat() ) //sparse input format
		{
			HashMap<CellIndex, Double> sparseMap = in.getSparseMap();
			for(Entry<CellIndex,Double> cell : sparseMap.entrySet() )
			{
				CellIndex key = cell.getKey();
				
				double value  = cell.getValue();  //input value
				if( value != 0 )                  //for all nnz
				{
					out.setValue(key, value);
				}
			}
		}
		else //dense input format
		{
			//for a merge this case will seldom happen, as each input MatrixObject
			//has at most 1/numThreads of all values in it.
			double[] values = in.getDenseArray();
			int rows = in.getNumRows();
			int cols = in.getNumColumns();
			
			for( int i=0; i<rows; i++ )
				for( int j=0; j<cols; j++ )
				{
				    double value = values[i*cols + j];  //input value
					if( value != 0  ) 					//for all nnz
					{
						out.setValue( i, j, value );	
					}
				}
		}	
	}
	
	
	/**
	 * 
	 */
	private class ResultMergeWorker implements Runnable
	{
		private MatrixBlock _inMB  = null;
		private MatrixBlock _outMB = null;
		
		public ResultMergeWorker(MatrixBlock inMB, MatrixBlock outMB)
		{
			_inMB  = inMB;
			_outMB = outMB;
		}

		@Override
		public void run() 
		{
			//read each input if required
			try
			{
				merge( _outMB, _inMB );
			}
			catch(Exception ex)
			{
				throw new RuntimeException("Failed to parallel merge result.", ex);
			}
		}
		
	}
}
