/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;

/**
 * This is a fused function object, introduced mainly for performance reasons
 * in order to condense binary operations to a unary operation.
 * Example usecase: logisticnew * (1 - logisticnew) in Logistic Regression.
 * 
 * TODO: As already discussed, we might want to generalize this to fused operators
 * for arbitrary expressions over a unary matrix input. However, for the moment, we
 * use this simplified function object in order to exploit block operations instead of 
 * instruction execution per single cell value that would be required if we want to 
 * support arbitary expressions. 
 * 
 */
public class Power2CMinus extends ValueFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static Power2CMinus singleObj = null;
	
	private Power2CMinus() {
		// nothing to do here
	}
	
	public static Power2CMinus getPower2CMFnObject() {
		if ( singleObj == null )
			singleObj = new Power2CMinus();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public double execute(double in1, double in2) {
		return in1*(in2-in1); //in2 is the passed constant 
	}

	@Override
	public double execute(double in1, long in2) {
		return in1*(in2-in1); //in2 is the passed constant
	}

	@Override
	public double execute(long in1, double in2) {
		return in1*(in2-in1); //in2 is the passed constant
	}

	@Override
	public double execute(long in1, long in2) {
		//for robustness regarding long overflows (only used for scalar instructions)
		double dval = ((double)in1*(in2-in1));
		if( dval > Long.MAX_VALUE )
			return dval;
		
		return in1*(in2-in1); //in2 is the passed constant
	}

}
