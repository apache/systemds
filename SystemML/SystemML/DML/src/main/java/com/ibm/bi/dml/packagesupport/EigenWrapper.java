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
import com.ibm.bi.dml.packagesupport.Scalar.ScalarType;
import com.ibm.bi.dml.packagesupport.PackageFunction;
import com.ibm.bi.dml.packagesupport.PackageRuntimeException;

/**
 * Wrapper class for eigen value computation
 * 
 * eigen = externalFunction(Matrix[Double] A) return (Matrix[Double] E,
 * Matrix[Double] V) implemented in
 * (classname="com.ibm.bi.dml.packagesupport.EigenWrapper",exectype="mem");
 * 
 * Support an additional int parameter, numComponents, for the number of
 * eigenpairs requested, and returning the number of eigenpairs computed.
 * 
 * eigen = externalFunction(Matrix[Double] A, Integer nC) return (Matrix[Double]
 * E, Matrix[Double] V, Integer nV) implemented in
 * (classname="com.ibm.bi.dml.packagesupport.EigenWrapper",exectype="mem");
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
	//eigen number of values
	private Scalar return_num_values;
	
	@Override
	public int getNumFunctionOutputs() 
	{
		if (this.getNumFunctionInputs()!= 2)
			return 2;
		else
			return 3;
	}

	@Override
	public FIO getFunctionOutput(int pos) 
	{	
		if(pos == 0)
			return return_e_values;
		
		if(pos == 1)
			return return_e_vectors;

		if(pos == 2 && this.getNumFunctionInputs() == 2)
			return return_num_values;
		
		throw new PackageRuntimeException("Invalid function output being requested");
	}

	@Override
	public void execute() 
	{ 
		Matrix input_m = (Matrix) this.getFunctionInput(0);
		double [][] arr = input_m.getMatrixAsDoubleArray();

		int numComponents = 0;
		if (this.getNumFunctionInputs()== 2)
			numComponents = Integer.parseInt(((Scalar)getFunctionInput(1)).getValue());
		
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
				if (this.getNumFunctionInputs() != 2)
					org.netlib.lapack.DSYEVR.DSYEVR("V", "A", "U", arr.length, arr, -1, -1, -1, -1, 0.0, numValues, e_values, e_vectors, i_suppz, work, 26*arr.length, iwork, 10*arr.length, info);
				else
					org.netlib.lapack.DSYEVR.DSYEVR("V", (numComponents==arr.length) ? "A":"I" , "U", arr.length, arr, -1, -1, 1, numComponents, 0.0, numValues, e_values, e_vectors, i_suppz, work, 26*arr.length, iwork, 10*arr.length, info);
			}
			
			/*double[][] tmp = new double[e_values.length][e_values.length];
			for(int i=0; i < e_values.length; i++)
				tmp[i][i] = e_values[i];*/
			
			//create output matrices
			return_e_values = new Matrix( e_values.length, e_values.length, ValueType.Double );
			return_e_values.setMatrixDoubleArray(e_values);			
			return_e_vectors = new Matrix( e_vectors.length, e_vectors.length, ValueType.Double );
			return_e_vectors.setMatrixDoubleArray(e_vectors);
			if (this.getNumFunctionInputs() == 2)
				return_num_values = new Scalar(ScalarType.Integer, String.valueOf(numValues.val));
	           
		}
		catch(Exception e)
		{
			throw new PackageRuntimeException("Error performing eigen",e);
		}
	}
}
