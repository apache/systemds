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
