/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.udf;

import java.io.Serializable;

/**
 * abstract class to represent all input and output objects for package
 * functions.
 * 
 * 
 * 
 */

public abstract class FunctionParameter implements Serializable
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	

	private static final long serialVersionUID = 1189133371204708466L;
	
	public enum FunctionParameterType{
		Matrix, 
		Scalar, 
		Object,
	}
	
	private FunctionParameterType _type;

	/**
	 * Constructor to set type
	 * 
	 * @param type
	 */
	public FunctionParameter(FunctionParameterType type) {
		_type = type;
	}

	/**
	 * Method to get type
	 * 
	 * @return
	 */
	public FunctionParameterType getType() {
		return _type;
	}

}
