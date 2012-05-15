/**
 * 
 */
package com.ibm.bi.dml.utils;

/**
 * <p>Exception occurring at the Language level.</p>
 * 
 * @author reinwald
 */
@SuppressWarnings("serial")
public class LanguageException extends DMLException {

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
    	public static String UNSUPPORTED_EXPRESSION = "Unsupported Expression";
    	public static String INVALID_PARAMETERS = "Invalid Parameters";
    	public static String UNSUPPORTED_PARAMETERS = "Unsupported Parameters";
    	public static String GENERIC_ERROR = "Language Syntax Error";
    }

}
