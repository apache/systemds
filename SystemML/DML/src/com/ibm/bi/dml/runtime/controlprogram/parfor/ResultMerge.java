package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.utils.DMLRuntimeException;

/**
 * Due to independence of all iterations, any result has the following properties:
 * (1) non local var, (2) matrix object, and (3) completely independent.
 * These properties allow us to realize result merging in parallel without any synchronization. 
 * 
 * 
 * RESTRICTIONS: 
 *   * assumption that each result matrix fits entirely in main-memory
 *   * currently no reduction aggregations functions
 */
public class ResultMerge 
{
	//inputs to result merge
	private MatrixObjectNew   _output      = null;
	private MatrixObjectNew[] _inputs      = null; 
	private String            _outputFName = null;
	private int               _par         = -1;
	
	
	//internal comparison matrix
	private double[][]        _compare     = null;
	
	public ResultMerge( MatrixObjectNew out, MatrixObjectNew[] in, String outputFilename, int par )
	{
		//System.out.println( "ResultMerge for output file "+out.getFileName()+", "+out.hashCode() );
		
		_output = out;
		_inputs = in;
		_outputFName = outputFilename;
		
		_par = par;
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
		return executeResultMerge( false );
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
	public MatrixObjectNew executeParallelMerge() 
		throws DMLRuntimeException
	{
		return executeResultMerge( true );
	}
	
	/**
	 * 
	 * @param inParallel
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private MatrixObjectNew executeResultMerge( boolean inParallel ) 
		throws DMLRuntimeException
	{
		MatrixObjectNew moNew = null; //always create new matrix object (required for nested parallelism)

		try
		{
			//get matrix blocks through caching 
			MatrixBlock outMB = _output.acquireRead();
			ArrayList<MatrixBlock> inMB = new ArrayList<MatrixBlock>();
			for( MatrixObjectNew in : _inputs )
			{
				//check for empty inputs (no iterations executed)
				if( in !=null && in != _output ) 
				{
					inMB.add( in.acquireRead() );	
				}
			}
			
			if( inMB.size() > 0 ) //if there exist something to merge
			{
				//get old output matrix from cache for compare
				MatrixBlock outMBNew = new MatrixBlock(outMB.getNumRows(), outMB.getNumColumns(), outMB.isInSparseFormat());

				//create compare matrix if required (existing data in result)
				_compare = createCompareMatrix(outMB);
				if( _compare != null )
					outMBNew.copy(outMB);
					
				//create new output matrix 
				String varname = _output.getVarName();
				moNew = new MatrixObjectNew(_output.getValueType(), _outputFName);
				moNew.setVarName( varname.contains("_rm") ? varname : varname+"_rm" );
				moNew.setDataType(DataType.MATRIX);
				moNew.setMetaData(createDeepCopyMetaData((MatrixFormatMetaData)_output.getMetaData()));
				
				//actual merge
				if( inParallel && !outMBNew.isInSparseFormat() ) //TODO remove degradation to serial once we synchronized internally
				{
					Timing time = new Timing();
					time.start();
					
					//parallel merge
					int numThreads = (_par > 0 && _par<inMB.size()) ? _par : Math.min(inMB.size(), InfrastructureAnalyzer.getLocalParallelism());
					Thread[] threads = new Thread[ numThreads ];
					//create and start threads
					for( int i=0; i<threads.length; i++ )
					{
						ResultMergeWorker rmw = new ResultMergeWorker(inMB.get(i), outMBNew);
						threads[i] = new Thread(rmw);
						threads[i].setPriority(Thread.MAX_PRIORITY);
						threads[i].start(); // start execution
					}	
					//wait for all workers to finish
					for( int i=0; i<threads.length; i++ )
					{
						threads[i].join();
					}
					
					System.out.println("PARALLEL RESULT MERGE with par="+inMB.size()+" in "+time.stop());
				}
				else
				{
					//serial merge
					for( MatrixBlock linMB : inMB )
						merge( outMBNew, linMB );
				}

				//release new output
				moNew.acquireModify(outMBNew);	
				moNew.release();
			}
			else
			{
				moNew = _output; //return old matrix, to prevent copy
			}
			
			//release old output, and all inputs
			_output.release();
			//_output.clearData(); //save, since it respects pin/unpin  
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

		return moNew;		
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

	private MatrixFormatMetaData createDeepCopyMetaData(MatrixFormatMetaData metaOld)
	{
		MatrixCharacteristics mcOld = metaOld.getMatrixCharacteristics();
		OutputInfo oiOld = metaOld.getOutputInfo();
		InputInfo iiOld = metaOld.getInputInfo();
		
		MatrixCharacteristics mc = new MatrixCharacteristics(mcOld.get_rows(),mcOld.get_cols(),
				                                             mcOld.get_rows_per_block(),mcOld.get_cols_per_block());
		MatrixFormatMetaData meta = new MatrixFormatMetaData(mc,oiOld,iiOld);
		return meta;
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
