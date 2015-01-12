/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import com.ibm.bi.dml.api.DMLException;

/**
 * <p>Exception occurring at the Language level.</p>
 */
public class LanguageException extends DMLException 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 1L;
	
    /**
     * @see java.lang.Exception#Exception()
     */
    public LanguageException() {
        super();
    }
    
    /**
     * @see java.lang.Exception#Exception(String)
     */
    public LanguageException(String message) {
        super(message);
    }
    
    /**
     * @see java.lang.Exception#Exception(Throwable)
     */
    public LanguageException(Throwable cause) {
        super(cause);
    }
    
    /**
     * @see java.lang.Exception#Exception(String, Throwable)
     */
    public LanguageException(String message, Throwable cause) {
        super(message, cause);
    }
    
    public LanguageException(String message, String code) {
        super(code + ERROR_MSG_DELIMITER + message);
    }
    
    public static class LanguageErrorCodes {
    	public static final String UNSUPPORTED_EXPRESSION = "Unsupported Expression";
    	public static final String INVALID_PARAMETERS = "Invalid Parameters";
    	public static final String UNSUPPORTED_PARAMETERS = "Unsupported Parameters";
    	public static final String GENERIC_ERROR = "Language Syntax Error";
    }

}
