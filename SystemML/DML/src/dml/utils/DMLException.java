package dml.utils;

/**
 * <p>Exception occurring in the DML framework.</p>
 * 
 * @author schnetter
 */
@SuppressWarnings("serial")
public abstract class DMLException extends Exception {

	public static String ERROR_MSG_DELIMITER = " : ";
	
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
