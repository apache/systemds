package com.ibm.bi.dml.packagesupport;

import java.util.Arrays;
import java.util.Comparator;

import com.ibm.bi.dml.packagesupport.Matrix;
import com.ibm.bi.dml.packagesupport.Matrix.ValueType;
import com.ibm.bi.dml.packagesupport.PackageFunction;
import com.ibm.bi.dml.packagesupport.PackageRuntimeException;
/**
 * Wrapper class for Order rows based on values in a column
 *
 */
public class OrderWrapper extends PackageFunction 
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
			int col = Integer.parseInt(((Scalar)getFunctionInput(1)).getValue());			
			boolean desc = Boolean.parseBoolean(((Scalar)getFunctionInput(2)).getValue());	
			
			//sort input matrix (in-place)
			if( !desc ) //asc
				Arrays.sort(inData, new AscRowComparator(col-1));
			else //desc
				Arrays.sort(inData, new DescRowComparator(Math.abs(col)-1));
				
			//create and copy output matrix
		
			String dir = createOutputFilePathAndName( OUTPUT_FILE );	
			ret = new Matrix( dir, inM.getNumRows(), inM.getNumCols(), ValueType.Double );
			ret.setMatrixDoubleArray(inData);
		} 
		catch (Exception e) 
		{
			throw new PackageRuntimeException("Error executing external order function", e);
		}
	}

	/**
	 * 
	 *
	 */
	private class AscRowComparator implements Comparator<double[]> 
	{
		private int _col = -1;
		
		public AscRowComparator( int col )
		{
			_col = col;
		}

		@Override
		public int compare(double[] arg0, double[] arg1) 
		{			
			return (arg0[_col] < arg1[_col] ? -1 : (arg0[_col] == arg1[_col] ? 0 : 1));
		}		
	}
	
	/**
	 * 
	 * 
	 */
	private class DescRowComparator implements Comparator<double[]> 
	{
		private int _col = -1;
		
		public DescRowComparator( int col )
		{
			_col = col;
		}

		@Override
		public int compare(double[] arg0, double[] arg1) 
		{			
			return (arg0[_col] > arg1[_col] ? -1 : (arg0[_col] == arg1[_col] ? 0 : 1));
		}		
	}
}
