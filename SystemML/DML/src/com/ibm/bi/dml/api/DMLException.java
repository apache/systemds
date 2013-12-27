/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.api;

/**
 * <p>Exception occurring in the DML framework.</p>
 */
public class DMLException extends Exception 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static String ERROR_MSG_DELIMITER = " : ";
	
	private static final long serialVersionUID = 1L;

	
    /**
     * @see java.lang.Exception#Exception()
     */
    public DMLException() {
        super();
    }
    
    /**
     * @see java.lang.Exception#Exception(String)
     */
    public DMLException(String message) {
        super(message);
    }
    
    /**
     * @see java.lang.Exception#Exception(Throwable)
     */
    public DMLException(Throwable cause) {
        super(cause);
    }
    
    /**
     * @see java.lang.Exception#Exception(String, Throwable)
     */
    public DMLException(String message, Throwable cause) {
        super(message, cause);
    }
}
