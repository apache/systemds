/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.udf.lib;

import org.apache.sysml.udf.FunctionParameter;
import org.apache.sysml.udf.Matrix;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.Scalar;
import org.apache.sysml.udf.Matrix.ValueType;
import org.apache.sysml.udf.Scalar.ScalarValueType;

/**
 * Wrapper class for binning a sorted input vector
 *
 */
public class BinningWrapper extends PackageFunction 
{
	
	private static final long serialVersionUID = 1L;
	private static final String OUTPUT_FILE = "TMP";

	//return matrix
	private Matrix _bins; //return matrix col_bins
	private Scalar _defBins; //return num defined bins

	@Override
	public int getNumFunctionOutputs() 
	{
		return 2;	
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) 
	{	
		switch(pos) {
			case 0: return _bins;
			case 1: return _defBins;
		}
		
		throw new RuntimeException("Invalid function output being requested");
	}

	@Override
	public void execute() 
	{ 
		try 
		{
			//get input parameters (input matrix assumed to be sorted)
			Matrix inM = (Matrix) getFunctionInput(0);
			double [][] col = inM.getMatrixAsDoubleArray();
			int binsize = Integer.parseInt(((Scalar)getFunctionInput(1)).getValue());	 
			int numbins = Integer.parseInt(((Scalar)getFunctionInput(2)).getValue());	 
			int nrowX = (int) inM.getNumRows();
			
			//execute binning (extend bins for duplicates)
			double[] col_bins = new double[numbins+1];
			int pos_col = 0;
			int bin_id = 0;
			
			col_bins[0] = col[0][0]; 
			while(pos_col < nrowX-1 && bin_id < numbins) { //for all bins
				pos_col = (pos_col + binsize >= nrowX) ? nrowX-1 : pos_col + binsize;
				double end_val = col[pos_col][0];
				col_bins[bin_id+1] = end_val;	
				
				//pull all duplicates in current bin
				boolean cont = true;
				while( cont && pos_col < nrowX-1 ){
					if( end_val == col[pos_col+1][0] )
			        	pos_col++;
				  	else 
				  		cont = false;
				}							
				bin_id++;
			}
			
			//prepare results
			int num_bins_defined = bin_id;
			for( int i=0; i<num_bins_defined; i++ )
				col_bins[i] = (col_bins[i] + col_bins[i+1])/2;
				
			//create and copy output matrix		
			String dir = createOutputFilePathAndName( OUTPUT_FILE );	
			_bins = new Matrix( dir, col_bins.length, 1, ValueType.Double );
			_bins.setMatrixDoubleArray(col_bins);
			_defBins = new Scalar(ScalarValueType.Integer, String.valueOf(num_bins_defined));
		} 
		catch (Exception e) 
		{
			throw new RuntimeException("Error executing external order function", e);
		}
	}
}
