/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;

// Singleton class

public class Power2 extends ValueFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = -4370611388912121328L;

	private static Power2 singleObj = null;
	
	private Power2() {
		// nothing to do here
	}
	
	public static Power2 getPower2FnObject() {
		if ( singleObj == null )
			singleObj = new Power2();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public double execute(double in1, double in2) {
		return in1*in1; //ignore in2 because always 2; 
	}

	@Override
	public double execute(double in1, long in2) {
		return in1*in1; //ignore in2 because always 2;
	}

	@Override
	public double execute(long in1, double in2) {
		return in1*in1; //ignore in2 because always 2;
	}

	@Override
	public double execute(long in1, long in2) {
		//for robustness regarding long overflows (only used for scalar instructions)
		double dval = ((double)in1 * in2);
		if( dval > Long.MAX_VALUE )
			return dval;	
		
		return in1*in1; //ignore in2 because always 2;
	}

}
