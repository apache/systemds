/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.udf.FunctionParameter;
import com.ibm.bi.dml.udf.Matrix;
import com.ibm.bi.dml.udf.PackageFunction;
import com.ibm.bi.dml.udf.PackageRuntimeException;
import com.ibm.bi.dml.udf.Matrix.ValueType;

/**
 * External function (type CP) used within the PerfTestTool in order to
 * measure the general behavior of package support.
 *
 */
public class PerfTestExtFunctCP extends PackageFunction 
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long   serialVersionUID = 1L;
	private static final String OUTPUT_FILE      = "PerfTestExtFunctOutput";
	
	private static IDSequence   _idSeq   = new IDSequence(); 
	private Matrix              _ret     = null; 
	private String              _baseDir = null;
	
	@Override
	public int getNumFunctionOutputs() 
	{
		return 1;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) 
	{
		return _ret;
	}
	
	public void setBaseDir(String dir)
	{
		_baseDir = dir;
	}

	@Override
	public void execute() 
	{
		try
		{
			long id = _idSeq.getNextID();
			
			Matrix in = (Matrix) this.getFunctionInput(0);
			double [][] aIn = in.getMatrixAsDoubleArray();
			
			int rows = aIn.length;
			int cols = aIn[0].length;
			
			String dir = _baseDir + "/" + OUTPUT_FILE+id;
			
			//copy and write output data 
			MatrixBlock mb = new MatrixBlock(rows,cols,false);
			for(int i=0; i < rows; i++)
				for(int j=0; j < cols; j++)
					mb.setValue(i, j, aIn[i][j]);

			_ret = new Matrix(dir, rows, cols, ValueType.Double);
			_ret.setMatrixDoubleArray(mb, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);	
		}
		catch(Exception e)
		{
			throw new PackageRuntimeException("Error executing generic test extfunct.", e);
		}
	}

}
