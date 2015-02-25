/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;


import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptimizerRuleBased;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixDimensionsMetaData;

/**
 * 
 */
public class ResultMergeLocalAutomatic extends ResultMerge
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private ResultMerge _rm = null;
	
	public ResultMergeLocalAutomatic( MatrixObject out, MatrixObject[] in, String outputFilename )
	{
		super( out, in, outputFilename );
	}

	@Override
	public MatrixObject executeSerialMerge() 
		throws DMLRuntimeException 
	{
		Timing time = new Timing(true);
		
		MatrixDimensionsMetaData metadata = (MatrixDimensionsMetaData) _output.getMetaData();
		MatrixCharacteristics mc = metadata.getMatrixCharacteristics();
		long rows = mc.getRows();
		long cols = mc.getCols();
		
		if( OptimizerRuleBased.isInMemoryResultMerge(rows, cols, OptimizerUtils.getLocalMemBudget()) )
			_rm = new ResultMergeLocalMemory( _output, _inputs, _outputFName );
		else
			_rm = new ResultMergeLocalFile( _output, _inputs, _outputFName );
		
		MatrixObject ret = _rm.executeSerialMerge();

		LOG.trace("Automatic result merge ("+_rm.getClass().getName()+") executed in "+time.stop()+"ms.");

		return ret;
	}
	
	@Override
	public MatrixObject executeParallelMerge(int par) 
		throws DMLRuntimeException 
	{
		MatrixDimensionsMetaData metadata = (MatrixDimensionsMetaData) _output.getMetaData();
		MatrixCharacteristics mc = metadata.getMatrixCharacteristics();
		long rows = mc.getRows();
		long cols = mc.getCols();
		
		if( OptimizerRuleBased.isInMemoryResultMerge(par * rows, cols, OptimizerUtils.getLocalMemBudget()) )
			_rm = new ResultMergeLocalMemory( _output, _inputs, _outputFName );
		else
			_rm = new ResultMergeLocalFile( _output, _inputs, _outputFName );
		
		return _rm.executeParallelMerge(par);	
	}
}
