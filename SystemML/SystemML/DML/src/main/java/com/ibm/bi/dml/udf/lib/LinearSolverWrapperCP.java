/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.udf.lib;

import org.netlib.util.intW;


import com.ibm.bi.dml.udf.FunctionParameter;
import com.ibm.bi.dml.udf.Matrix;
import com.ibm.bi.dml.udf.PackageFunction;
import com.ibm.bi.dml.udf.PackageRuntimeException;
import com.ibm.bi.dml.udf.Matrix.ValueType;


/**
 * 
 *
 */
public class LinearSolverWrapperCP extends PackageFunction 
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 1L;
	private static final String OUTPUT_FILE = "TMP";

	private Matrix _ret; 
		
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
	
	@Override
	public void execute() 
	{
		try
		{
			Matrix mA = (Matrix) this.getFunctionInput(0);
			Matrix mB = (Matrix) this.getFunctionInput(1);
			
			double[][] aA = mA.getMatrixAsDoubleArray();
			double[][] aB = mB.getMatrixAsDoubleArray();
			
			int rows = aA.length;
			int cols = aB[0].length;
			intW tmp = new intW(0);
			
			org.netlib.lapack.DGESV.DGESV(rows, cols, aA, new int[rows], aB, tmp);
			
			if( tmp.val != 0 ) //eval error codes
			{
				String msg = null;
				if( tmp.val<0 ) 
				{
					msg = "the "+Math.abs(tmp.val)+"-th argument had an illegal value";
				}
				else if ( tmp.val>0 ) 
				{
					msg = "U("+tmp.val+","+tmp.val+") is exactly zero. " +
						  "The factorization has been completed, but the factor U is exactly singular, " +
						  "so the solution could not be computed.";
				}											
				LOG.warn( msg );
			}

			//create and copy output data 
			String dir = createOutputFilePathAndName( OUTPUT_FILE );			
			_ret = new Matrix( dir, aB.length, aB[0].length, ValueType.Double);
			_ret.setMatrixDoubleArray( aB );
		}
		catch(Exception e)
		{
			throw new PackageRuntimeException("Error executing external Lapack function", e);
		}
	}

}
