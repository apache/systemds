/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.udf.lib;

import com.ibm.bi.dml.udf.FunctionParameter;
import com.ibm.bi.dml.udf.Matrix;
import com.ibm.bi.dml.udf.PackageFunction;
import com.ibm.bi.dml.udf.PackageRuntimeException;
import com.ibm.bi.dml.udf.Scalar;
import com.ibm.bi.dml.udf.Matrix.ValueType;
import com.ibm.bi.dml.udf.Scalar.ScalarValueType;

/**
 * Wrapper class for binning a sorted input vector
 *
 */
public class BinningWrapper extends PackageFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 1L;
	private final String OUTPUT_FILE = "TMP";

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
		
		throw new PackageRuntimeException("Invalid function output being requested");
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
			throw new PackageRuntimeException("Error executing external order function", e);
		}
	}
}
