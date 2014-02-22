/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;

import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class IntegerDivide extends ValueFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static IntegerDivide singleObj = null;
	
	private IntegerDivide() {
		// nothing to do here
	}
	
	public static IntegerDivide getIntegerDivideFnObject() {
		if ( singleObj == null )
			singleObj = new IntegerDivide();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public double execute(double in1, double in2) {
		return executeIntDiv( in1, in2 );
	}

	@Override
	public double execute(double in1, int in2) {
		return executeIntDiv( in1, (double)in2 );
	}

	@Override
	public double execute(int in1, double in2) {
		return executeIntDiv( (double)in1, in2 );
	}

	@Override
	public double execute(int in1, int in2) {
		return executeIntDiv( (double)in1, (double)in2 );
	}

	/**
	 * NOTE: The R semantics of integer divide a%/%b are to compute the 
	 * double division and subsequently cast to int. In case of a NaN 
	 * or +-INFINITY devision result, the overall output is NOT cast to
	 * int in order to prevent the special double values.
	 * 
	 * @param in1
	 * @param in2
	 * @return
	 */
	private double executeIntDiv( double in1, double in2 )
	{
		//compute normal double devision
		double ret = in1 / in2;
		
		//check for NaN/+-INF intermediate (cast to int would eliminate it)
		if( Double.isNaN(ret) || Double.isInfinite(ret) )
			return ret;
		
		//safe cast to int
		return UtilFunctions.toInt( ret );
	}
}
