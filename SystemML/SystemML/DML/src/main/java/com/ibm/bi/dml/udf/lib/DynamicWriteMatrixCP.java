/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.udf.lib;

import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixDimensionsMetaData;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.udf.FunctionParameter;
import com.ibm.bi.dml.udf.Matrix;
import com.ibm.bi.dml.udf.PackageFunction;
import com.ibm.bi.dml.udf.PackageRuntimeException;
import com.ibm.bi.dml.udf.Scalar;
import com.ibm.bi.dml.udf.Scalar.ScalarValueType;

/**
 * 
 *
 */
public class DynamicWriteMatrixCP extends PackageFunction 
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 1L;
	private Scalar _success;
	
	@Override
	public int getNumFunctionOutputs() 
	{
		return 1;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) 
	{
		return _success;
	}

	@Override
	public void execute() 
	{
		boolean success = false;
		
		try
		{
			Matrix mat = (Matrix) this.getFunctionInput(0);
			String fname = ((Scalar) this.getFunctionInput(1)).getValue();
			String format = ((Scalar) this.getFunctionInput(2)).getValue();
			
			MatrixObject mo = mat.getMatrixObject();
			MatrixCharacteristics mc = ((MatrixDimensionsMetaData)mo.getMetaData()).getMatrixCharacteristics();
			OutputInfo oi = OutputInfo.stringToOutputInfo(format);
			
			MatrixBlock mb = mo.acquireRead();
			DataConverter.writeMatrixToHDFS(mb, fname, oi, mc);			
			mo.release();
			
			success = true;
		}
		catch(Exception e)
		{
			throw new PackageRuntimeException("Error executing dynamic write of matrix",e);
		}
		
		_success = new Scalar(ScalarValueType.Boolean,String.valueOf(success));
	}
}
