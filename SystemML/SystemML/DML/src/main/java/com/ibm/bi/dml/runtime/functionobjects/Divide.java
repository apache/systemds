/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;

public class Divide extends ValueFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static Divide singleObj = null;
	
	private Divide() {
		// nothing to do here
	}
	
	public static Divide getDivideFnObject() {
		if ( singleObj == null )
			singleObj = new Divide();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public double execute(double in1, double in2) {
		return in1 / in2;
	}

	@Override
	public double execute(double in1, long in2) {
		return in1 / (double)in2;
	}

	@Override
	public double execute(long in1, double in2) {
		return (double)in1 / in2;
	}

	@Override
	public double execute(long in1, long in2) {
		return (double)in1 / (double)in2;
	}

}
