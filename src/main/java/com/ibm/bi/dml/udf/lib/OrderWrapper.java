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

package com.ibm.bi.dml.udf.lib;

import java.util.Arrays;
import java.util.Comparator;

import com.ibm.bi.dml.udf.FunctionParameter;
import com.ibm.bi.dml.udf.Matrix;
import com.ibm.bi.dml.udf.PackageFunction;
import com.ibm.bi.dml.udf.PackageRuntimeException;
import com.ibm.bi.dml.udf.Scalar;
import com.ibm.bi.dml.udf.Matrix.ValueType;
/**
 * Wrapper class for Order rows based on values in a column
 *
 */
public class OrderWrapper extends PackageFunction 
{
	
	private static final long serialVersionUID = 1L;
	private static final String OUTPUT_FILE = "TMP";

	//return matrix
	private Matrix ret;

	@Override
	public int getNumFunctionOutputs() 
	{
		return 1;	
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) 
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
				Arrays.sort(inData, new DescRowComparator(col-1));
				
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
	private static class AscRowComparator implements Comparator<double[]> 
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
	private static class DescRowComparator implements Comparator<double[]> 
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
