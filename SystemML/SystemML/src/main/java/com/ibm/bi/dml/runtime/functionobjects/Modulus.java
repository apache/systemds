/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;

/**
 * Integer modulus, where we adhere to the defined R semantics:
 * 
 * ("%% indicates x mod y and %/% indicates integer division. 
 * It is guaranteed that x == (x %% y) + y * ( x %/% y ) (up to rounding error) 
 * unless y == 0")
 * 
 */
public class Modulus extends ValueFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private static Modulus singleObj = null;
	private IntegerDivide _intdiv = null;
	
	private Modulus() {
		_intdiv = IntegerDivide.getIntegerDivideFnObject();
	}
	
	public static Modulus getModulusFnObject() {
		if ( singleObj == null )
			singleObj = new Modulus();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public double execute(double in1, double in2) {
		if( in2==0.0 || in2==-0.0 )
			return Double.NaN;
		return in1 - _intdiv.execute(in1, in2)*in2;
	}

	@Override
	public double execute(double in1, long in2) {
		if( in2==0 )
			return Double.NaN;
		return in1 - _intdiv.execute(in1, in2)*in2;
	}

	@Override
	public double execute(long in1, double in2) {
		if( in2==0.0 || in2==-0.0 )
			return Double.NaN;
		return in1 - _intdiv.execute(in1, in2)*in2;
	}

	@Override
	public double execute(long in1, long in2) {
		if( in2==0 )
			return Double.NaN;
		return in1 - _intdiv.execute(in1, in2)*in2;
	}	
	
}
