/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.udf;

/**
 * Class to represent a scalar input/output.
 * 
 * 
 * 
 */
public class Scalar extends FunctionParameter 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	

	private static final long serialVersionUID = 55239661026793046L;

	public enum ScalarValueType {
		Integer, Double, Boolean, Text
	};

	protected String _value;
	protected ScalarValueType _sType;

	/**
	 * Constructor to setup a scalar object.
	 * 
	 * @param t
	 * @param val
	 */
	public Scalar(ScalarValueType t, String val) {
		super(FunctionParameterType.Scalar);
		_sType = t;
		_value = val;
	}

	/**
	 * Method to get type of scalar.
	 * 
	 * @return
	 */
	public ScalarValueType getScalarType() {
		return _sType;
	}

	/**
	 * Method to get value for scalar.
	 * 
	 * @return
	 */
	public String getValue() {
		return _value;
	}

}
