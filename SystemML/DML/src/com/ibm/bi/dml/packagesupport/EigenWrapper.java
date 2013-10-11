/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.packagesupport;

import org.netlib.util.intW;
import com.ibm.bi.dml.packagesupport.FIO;
import com.ibm.bi.dml.packagesupport.Matrix;
import com.ibm.bi.dml.packagesupport.Matrix.ValueType;
import com.ibm.bi.dml.packagesupport.PackageFunction;
import com.ibm.bi.dml.packagesupport.PackageRuntimeException;
/**
 * Wrapper class for eigen value computation
 *
 */
public class EigenWrapper extends PackageFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 87925877955636432L;
	
	//eigen values matrix
	private Matrix return_e_values;
	//eigen vectors matrix
	private Matrix return_e_vectors; 

	@Override
	public int getNumFunctionOutputs() 
	{
		return 2;	
	}

	@Override
	public FIO getFunctionOutput(int pos) 
	{	
		if(pos == 0)
			return return_e_values;
		
		if(pos == 1)
			return return_e_vectors;
		
		throw new PackageRuntimeException("Invalid function output being requested");
	}

	@Override
	public void execute() 
	{ 
		Matrix input_m = (Matrix) this.getFunctionInput(0);
		double [][] arr = input_m.getMatrixAsDoubleArray();

		intW numValues = new intW(0); 
		intW info = new intW(0);
		double []e_values = new double[arr.length];
		double [][]e_vectors = new double[arr.length][arr.length];
		int[] i_suppz = new int[2* arr.length];
		double []work = new double[26*arr.length];
		int []iwork = new int[10*arr.length];
		
		try
		{
			// prevent multiple concurrent invocations in same JVM because: 
			// Lapack DSYEVR can run into deadlocks / sideeffects
			synchronized( org.netlib.lapack.DSYEVR.class )
			{
				//compute eigen values and vectors
				org.netlib.lapack.DSYEVR.DSYEVR("V", "A", "U", arr.length, arr, -1, -1, -1, -1, 0.0, numValues, e_values, e_vectors, i_suppz, work, 26*arr.length, iwork, 10*arr.length, info);
			}
			
			double[][] tmp = new double[e_values.length][e_values.length];
			for(int i=0; i < e_values.length; i++)
				tmp[i][i] = e_values[i];
			
			//create output matrices
			return_e_values = new Matrix( e_values.length, e_values.length, ValueType.Double );
			return_e_values.setMatrixDoubleArray(tmp);			
			return_e_vectors = new Matrix( e_vectors.length, e_vectors.length, ValueType.Double );
			return_e_vectors.setMatrixDoubleArray(e_vectors);			
		}
		catch(Exception e)
		{
			throw new PackageRuntimeException("Error performing eigen",e);
		}
	}
}
