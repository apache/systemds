package com.ibm.bi.dml.packagesupport;


import com.ibm.bi.dml.packagesupport.Matrix;
import com.ibm.bi.dml.packagesupport.Matrix.ValueType;
import com.ibm.bi.dml.packagesupport.PackageFunction;
import com.ibm.bi.dml.packagesupport.PackageRuntimeException;

/**
 * Wrapper class to deNaN matrices by replacing all NaNs with zeros,
 * made by modifying <code>OrderWrapper.java</code>
 */
public class DeNaNWrapper extends PackageFunction 
{
	private static final long serialVersionUID = 1L;
	private final String OUTPUT_FILE = "TMP";

	//return matrix
	private Matrix ret;

	@Override
	public int getNumFunctionOutputs() 
	{
		return 1;	
	}

	@Override
	public FIO getFunctionOutput(int pos) 
	{	
		if(pos == 0)
			return ret;
		
		throw new PackageRuntimeException("Invalid function output being requested");
	}

	@Override
	public void execute() 
	{ 
		try 
		{
			Matrix inM = (Matrix) getFunctionInput(0);
			double [][] inData = inM.getMatrixAsDoubleArray();
			for (int i = 0; i < inData.length; i++) {
				for (int j = 0; j < inData[i].length; j++) {
					if (Double.isNaN (inData [i][j])) {
						inData [i][j] = 0.0;
			}   }   }
			//create and copy output matrix		
			String dir = createOutputFilePathAndName( OUTPUT_FILE );	
			ret = new Matrix( dir, inM.getNumRows(), inM.getNumCols(), ValueType.Double );
			ret.setMatrixDoubleArray(inData);
		} 
		catch (Exception e) 
		{
			throw new PackageRuntimeException("Error executing external removeNaN function", e);
		}
	}
}
