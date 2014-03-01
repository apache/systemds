/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;

// Singleton class

public class Power extends ValueFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static Power singleObj = null;
	
	private Power() {
		// nothing to do here
	}
	
	public static Power getPowerFnObject() {
		if ( singleObj == null )
			singleObj = new Power();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public double execute(double in1, double in2) {
		return Math.pow(in1, in2); 
	}

	@Override
	public double execute(double in1, int in2) {
		return Math.pow(in1, (double)in2); 
	}

	@Override
	public double execute(int in1, double in2) {
		return Math.pow((double)in1, in2); 
	}

	@Override
	public double execute(int in1, int in2) {
		return Math.pow((double)in1, (double)in2);
	}

}
