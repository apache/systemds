/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime;

import com.ibm.bi.dml.api.DMLException;

/**
 * This exception should be thrown to flag DML Script errors.
 */
public class DMLScriptException extends DMLException 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 1L;

	public DMLScriptException(String string) {
		super(string);
	}
	
	public DMLScriptException(Exception e) {
		super(e);
	}

	public DMLScriptException(String string, Exception ex){
		super(string,ex);
	}
	
	public String prepErrorMessage() {
		return (this.getMessage().substring(DMLScriptException.class.toString().length()-4));
	}
}
