package com.ibm.bi.dml.packagesupport;

import org.netlib.util.intW;


import com.ibm.bi.dml.packagesupport.FIO;
import com.ibm.bi.dml.packagesupport.Matrix;
import com.ibm.bi.dml.packagesupport.PackageFunction;
import com.ibm.bi.dml.packagesupport.PackageRuntimeException;
import com.ibm.bi.dml.packagesupport.Matrix.ValueType;


/**
 * 
 *
 */
public class LinearSolverWrapperCP extends PackageFunction 
{	
	private static final long serialVersionUID = 1L;
	private final String OUTPUT_FILE = "TMP";

	private Matrix _ret; 
		
	@Override
	public int getNumFunctionOutputs() 
	{
		return 1;
	}

	@Override
	public FIO getFunctionOutput(int pos) 
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
				System.out.println("Warning: "+msg);
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
