/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.udf;

/**
 * Class to represent an object.
 * 
 */

public class BinaryObject extends FunctionParameter 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = 314464073593116450L;
	protected Object _o;

	/**
	 * constructor that takes object as param
	 * 
	 * @param o
	 */
	public BinaryObject(Object o) {
		super( FunctionParameterType.Object );
		_o = o;
	}

	/**
	 * Method to retrieve object.
	 * 
	 * @return
	 */
	public Object getObject() {
		return _o;
	}

}
