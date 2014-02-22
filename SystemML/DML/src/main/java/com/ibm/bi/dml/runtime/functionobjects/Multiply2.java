/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;

public class Multiply2 extends ValueFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static Multiply2 singleObj = null;
	
	private Multiply2() {
		// nothing to do here
	}
	
	public static Multiply2 getMultiply2FnObject() {
		if ( singleObj == null )
			singleObj = new Multiply2();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public double execute(double in1, double in2) {
		return in1 + in1; //ignore in2 because always 2; 
	}

	@Override
	public double execute(double in1, int in2) {
		return in1 + in1; //ignore in2 because always 2; 
	}

	@Override
	public double execute(int in1, double in2) {
		return in1 + in1; //ignore in2 because always 2; 
	}

	@Override
	public double execute(int in1, int in2) {
		return in1 + in1; //ignore in2 because always 2; 
	}

}
