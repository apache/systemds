/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.packagesupport;

/**
 * Class to represent a scalar input/output.
 * 
 * 
 * 
 */
public class Scalar extends FIO 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	

	private static final long serialVersionUID = 55239661026793046L;

	public enum ScalarType {
		Integer, Double, Boolean, Text
	};

	String value;
	ScalarType sType;

	/**
	 * Constructor to setup a scalar object.
	 * 
	 * @param t
	 * @param val
	 */
	public Scalar(ScalarType t, String val) {
		super(Type.Scalar);
		sType = t;
		value = val;
	}

	/**
	 * Method to get type of scalar.
	 * 
	 * @return
	 */
	public ScalarType getScalarType() {
		return sType;
	}

	/**
	 * Method to get value for scalar.
	 * 
	 * @return
	 */
	public String getValue() {
		return value;
	}

}
