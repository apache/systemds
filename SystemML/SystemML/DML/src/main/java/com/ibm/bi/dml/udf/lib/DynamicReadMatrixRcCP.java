/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.udf.lib;

import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.udf.FunctionParameter;
import com.ibm.bi.dml.udf.Matrix;
import com.ibm.bi.dml.udf.PackageFunction;
import com.ibm.bi.dml.udf.PackageRuntimeException;
import com.ibm.bi.dml.udf.Scalar;
import com.ibm.bi.dml.udf.Matrix.ValueType;
import com.ibm.bi.dml.udf.Scalar.ScalarValueType;

/**
 * 
 *
 */
public class DynamicReadMatrixRcCP extends PackageFunction 
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 1L;
	private Matrix _ret;
	private Scalar _rc;
	
	@Override
	public int getNumFunctionOutputs() 
	{
		return 2;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) 
	{
		if(pos == 0)
			return _ret;
		
		if(pos == 1)
			return _rc;
		
		throw new PackageRuntimeException("Invalid function output being requested");
		
	}

	@Override
	public void execute() 
	{
		try
		{
			String fname = ((Scalar) this.getFunctionInput(0)).getValue();
			Integer m = Integer.parseInt(((Scalar) this.getFunctionInput(1)).getValue());
			Integer n = Integer.parseInt(((Scalar) this.getFunctionInput(2)).getValue());
			String format = ((Scalar) this.getFunctionInput(3)).getValue();
			
			InputInfo ii = InputInfo.stringToInputInfo(format);
			OutputInfo oi = OutputInfo.BinaryBlockOutputInfo;
			
			String fnameTmp = createOutputFilePathAndName("TMP");
			_ret = new Matrix(fnameTmp, m, n, ValueType.Double);

			MatrixBlock mbTmp = DataConverter.readMatrixFromHDFS(fname, ii, m, n, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);			
			_ret.setMatrixDoubleArray(mbTmp, oi, ii);
			_rc = new Scalar(ScalarValueType.Integer, "0");
			
			//NOTE: The packagesupport wrapper creates a new MatrixObjectNew with the given 
			// matrix block. This leads to a dirty state of the new object. Hence, the resulting 
			// intermediate plan variable will be exported in front of MR jobs and during this export 
			// the format will be changed to binary block (the contract of external functions), 
			// no matter in which format the original matrix was.
		}
		catch(Exception e)
		{
			_rc = new Scalar(ScalarValueType.Integer, "1");
//			throw new PackageRuntimeException("Error executing dynamic read of matrix",e);
		}
	}
}
