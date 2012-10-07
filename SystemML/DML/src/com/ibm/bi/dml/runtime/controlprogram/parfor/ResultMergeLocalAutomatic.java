package com.ibm.bi.dml.runtime.controlprogram.parfor;


import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixDimensionsMetaData;
import com.ibm.bi.dml.utils.DMLRuntimeException;

/**
 * 
 */
public class ResultMergeLocalAutomatic extends ResultMerge
{
	private ResultMerge _rm = null;
	
	public ResultMergeLocalAutomatic( MatrixObject out, MatrixObject[] in, String outputFilename )
	{
		super( out, in, outputFilename );
	}

	@Override
	public MatrixObject executeSerialMerge() 
		throws DMLRuntimeException 
	{
		Timing time = new Timing();
		time.start();
		
		MatrixDimensionsMetaData metadata = (MatrixDimensionsMetaData) _output.getMetaData();
		MatrixCharacteristics mc = metadata.getMatrixCharacteristics();
		long rows = mc.get_rows();
		long cols = mc.get_cols();
		
		if( rows*cols < Math.pow(Hops.CPThreshold,2) )
			_rm = new ResultMergeLocalMemory( _output, _inputs, _outputFName );
		else
			_rm = new ResultMergeLocalFile( _output, _inputs, _outputFName );
		
		MatrixObject ret = _rm.executeSerialMerge();

		if( LDEBUG )
			System.out.println("Automatic result merge ("+_rm.getClass().getName()+") executed in "+time.stop()+"ms.");

		return ret;
	}
	
	@Override
	public MatrixObject executeParallelMerge(int par) 
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
