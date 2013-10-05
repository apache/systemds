/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;

import static com.ibm.bi.dml.runtime.util.UtilFunctions.toInt;

public class Modulus extends ValueFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private static final Modulus INSTANCE = new Modulus();
	
	private Modulus() {
		// nothing to do here
	}
	
	public static Modulus getModulusFnObject() {
		return INSTANCE;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public double execute(double in1, double in2) {
		return modOrNaN(toInt(in1), toInt(in2));
	}

	@Override
	public double execute(double in1, int in2) {
		return modOrNaN(toInt(in1), in2);
	}

	@Override
	public double execute(int in1, double in2) {
		return modOrNaN(in1, toInt(in2));
	}

	@Override
	public double execute(int in1, int in2) {
		return modOrNaN(in1, in2);
	}	
	
	public static double modOrNaN(int a, int b) {
		try {
			return a % b;
		} catch (ArithmeticException e) {
			// return NaN for mod 0
			return Double.NaN;
		}
	}

}
