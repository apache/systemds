package dml.runtime.controlprogram.parfor;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map.Entry;

import dml.runtime.instructions.CPInstructions.MatrixObject;
import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.MatrixFormatMetaData;
import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.runtime.util.DataConverter;
import dml.utils.DMLRuntimeException;

/**
 * Due to independence of all iterations, any results has the following properties:
 * (1) non local var, (2) matrix object, and (3) completely independent.
 * These properties allow us to realize result merging in parallel without any synchronization. 
 * 
 * RESTRICTIONS: 
 *   * assumption that each result matrix fits entirely in main-memory
 *   * currently no reduction aggregations functions
 * 
 * 
 * @author mboehm
 */
public class ResultMerge 
{
	private MatrixObject    _output = null;
	private MatrixObject[]  _inputs = null; 

	private boolean         _writeOutput = false;
	private boolean         _readInputs = false;
	
	public ResultMerge( MatrixObject out, MatrixObject[] in, boolean writeOutput, boolean readInputs )
	{
		_output = out;
		_inputs = in;
		
		_writeOutput = writeOutput;
		_readInputs = readInputs;
	}
	
	/**
	 * 
	 * @return
	 * @throws DMLRuntimeException
	 */
	public MatrixObject executeSerialMerge() 
		throws DMLRuntimeException
	{
		try
		{
			for( MatrixObject in : _inputs ) 
			{
				//read each input if required
				if( _readInputs )
					readMatrixObject( in );
				
				//merge each input to output
				merge( _output, in );
			}
			
			//write output if required
			if( _writeOutput )
				writeMatrixObject( _output );
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		
		return _output;
	}
	
	/**
	 * 
	 * @param par
	 * @return
	 * @throws DMLRuntimeException
	 */
	public MatrixObject executeParallelMerge(int par) 
		throws DMLRuntimeException
	{
		try
		{
			//create and start threads
			Thread[] threads = new Thread[ _inputs.length ];
			for( int i=0; i<threads.length; i++ )
			{
				ResultMergeWorker rmw = new ResultMergeWorker(_inputs[i], _output);
				threads[i] = new Thread(rmw);
				threads[i].start(); // start execution
			}
				
			//wait for all workers to finish
			for( int i=0; i<threads.length; i++ )
			{
				threads[i].join();
			}
			
			//write output if required
			if( _writeOutput )
				writeMatrixObject( _output );
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		
		return _output;
	}
	
	/**
	 * NOTE: similar to converters, but not directly applicable as we are interested in combining
	 * to objects with each other; not unary transformation.
	 * 
	 * @param in
	 * @return
	 */
	public MatrixObject merge( MatrixObject out, MatrixObject in )
	{
		MatrixBlock mbout = out.getData();
		MatrixBlock mbin = in.getData();
		
		
		if( mbin.isInSparseFormat() ) //sparse format
		{
			HashMap<CellIndex, Double> sparseMap = mbin.getSparseMap();
			for(Entry<CellIndex,Double> cell : sparseMap.entrySet() )
			{
				CellIndex key = cell.getKey();
				double value  = cell.getValue();
				
				mbout.setValue(key, value);
			}
		}
		else //dense format
		{
			//for a merge this case will seldom happen, as each input MatrixObject
			//has at most 1/numThreads of all values in it.
			double[] values = mbin.getDenseArray();
			int rows = mbin.getNumRows();
			int cols = mbin.getNumColumns();
			
			for( int i=0; i<rows; i++ )
				for( int j=0; j<cols; j++ )
					mbout.setValue(i, j, values[i*cols + j]);	
		}
		
		return _output;
	}
	
	/**
	 * 
	 * @param mo
	 * @throws IOException
	 */
	private void readMatrixObject( MatrixObject mo ) 
		throws IOException
	{
		String dir = mo.getFileName();
		
		MatrixFormatMetaData md = (MatrixFormatMetaData) mo.getMetaData();
		MatrixCharacteristics mc = md.getMatrixCharacteristics();
		
		MatrixBlock mb = DataConverter.readMatrixFromHDFS(dir, md.getInputInfo(), 
				                                          mc.get_rows(), mc.get_cols(), 
				                                          mc.get_rows_per_block(), mc.get_cols_per_block());
		mo.setData(mb);
	}
	
	/**
	 * 
	 * @param mo
	 * @throws IOException
	 */
	private void writeMatrixObject( MatrixObject mo ) 
		throws IOException 
	{
		MatrixBlock mb = mo.getData();
		String dir = mo.getFileName();
		
		MatrixFormatMetaData md = (MatrixFormatMetaData) mo.getMetaData();
		MatrixCharacteristics mc = md.getMatrixCharacteristics();
		
		//overwrite existing file
		DataConverter.writeMatrixToHDFS( mb, dir, md.getOutputInfo(), 
                                         mc.get_rows(), mc.get_cols(), 
                                         mc.get_rows_per_block(), mc.get_cols_per_block());	
	}
	
	/**
	 * 
	 * @author mboehm
	 */
	private class ResultMergeWorker implements Runnable
	{
		private MatrixObject _input  = null;
		private MatrixObject _output = null;
		
		public ResultMergeWorker(MatrixObject in, MatrixObject out)
		{
			_input  = in;
			_output = out;
		}

		@Override
		public void run() 
		{
			//read each input if required
			try
			{
				if( _readInputs )
					readMatrixObject( _input );
				
				merge( _output, _input );
			}
			catch(Exception ex)
			{
				throw new RuntimeException("Failed to parallel merge result.", ex);
			}
		}
		
	}
}
