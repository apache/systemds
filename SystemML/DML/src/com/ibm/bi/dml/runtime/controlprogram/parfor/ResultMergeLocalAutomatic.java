package com.ibm.bi.dml.runtime.controlprogram.parfor;


import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixDimensionsMetaData;
import com.ibm.bi.dml.utils.DMLRuntimeException;

/**
 * 
 */
public class ResultMergeLocalAutomatic extends ResultMerge
{
	private ResultMerge _rm = null;
	
	public ResultMergeLocalAutomatic( MatrixObjectNew out, MatrixObjectNew[] in, String outputFilename )
	{
		super( out, in, outputFilename );
	}

	@Override
	public MatrixObjectNew executeSerialMerge() 
		throws DMLRuntimeException 
	{
		MatrixDimensionsMetaData metadata = (MatrixDimensionsMetaData) _output.getMetaData();
		MatrixCharacteristics mc = metadata.getMatrixCharacteristics();
		long rows = mc.get_rows();
		long cols = mc.get_cols();
		
		if( rows*cols < Math.pow(Hops.CPThreshold,2) )
			_rm = new ResultMergeLocalMemory( _output, _inputs, _outputFName );
		else
			_rm = new ResultMergeLocalFile( _output, _inputs, _outputFName );
		
		return _rm.executeSerialMerge();
	}
	
	@Override
	public MatrixObjectNew executeParallelMerge(int par) 
		throws DMLRuntimeException 
	{
		MatrixDimensionsMetaData metadata = (MatrixDimensionsMetaData) _output.getMetaData();
		MatrixCharacteristics mc = metadata.getMatrixCharacteristics();
		long rows = mc.get_rows();
		long cols = mc.get_cols();
		
		if( par*(rows*cols) < Math.pow(Hops.CPThreshold,2) )
			_rm = new ResultMergeLocalMemory( _output, _inputs, _outputFName );
		else
			_rm = new ResultMergeLocalFile( _output, _inputs, _outputFName );
		
		return _rm.executeParallelMerge(par);	
	}
}
