/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime;


/**
 * This exception should be thrown to flag DML Script errors.
 */
public class DMLScriptException extends DMLRuntimeException 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 1L;

	//prevent string concatenation of classname w/ stop message
	private DMLScriptException(Exception e) {
		super(e);
	}

	private DMLScriptException(String string, Exception ex){
		super(string,ex);
	}
	
	/**
	 * This is the only valid constructor for DMLScriptException.
	 * 
	 * @param string
	 */
	public DMLScriptException(String msg) {
		super(msg);
	}
}
