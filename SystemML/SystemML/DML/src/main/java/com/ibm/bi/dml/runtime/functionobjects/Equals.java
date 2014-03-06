/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;

public class Equals extends ValueFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static Equals singleObj = null;

	private Equals() {
		// nothing to do here
	}
	
	public static Equals getEqualsFnObject() {
		if ( singleObj == null )
			singleObj = new Equals();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	/*
	 * Arithmetic relational operators (==, !=, <=, >=) must be instead of
	 * <code>Double.compare()</code> due to the inconsistencies in the way
	 * NaN and -0.0 are handled. The behavior of methods in
	 * <code>Double</code> class are designed mainly to make Java
	 * collections work properly. For more details, see the help for
	 * <code>Double.equals()</code> and <code>Double.comapreTo()</code>.
	 */
	
	/**
	 * execute() method that returns double is required since current map-reduce
	 * runtime can only produce matrices of doubles. This method is used on MR
	 * side to perform comparisons on matrices like A==B and A==2.5
	 */
	@Override
	public double execute(double in1, double in2) {
		return (in1 == in2 ? 1.0 : 0.0);
	}
	
	@Override
	public boolean compare(boolean in1, boolean in2) {
		return (in1 == in2);
	}

	@Override
	public boolean compare(double in1, double in2) {
		return (in1 == in2);
	}

	@Override
	public boolean compare(int in1, int in2) {
		return (in1 == in2);
	}

	@Override
	public boolean compare(double in1, int in2) {
		return (in1 == in2);
	}

	@Override
	public boolean compare(int in1, double in2) {
		return (in1 == in2);
	}
}
