/**
 * 
 */
package com.ibm.bi.dml.utils;

/**
 * <p>Exception occurring in the HOP level.</p>
 * 
 * @author reinwald
 */
@SuppressWarnings("serial")
public class HopsException extends DMLException {

    /**
     * @see java.lang.Exception#Exception()
     */
    public HopsException() {
        super();
    }
    
    /**
     * @see java.lang.Exception#Exception(String)
     */
    public HopsException(String message) {
        super(message);
    }
    
    /**
     * @see java.lang.Exception#Exception(Throwable)
     */
    public HopsException(Throwable cause) {
        super(cause);
    }
    
    /**
     * @see java.lang.Exception#Exception(String, Throwable)
     */
    public HopsException(String message, Throwable cause) {
        super(message, cause);
    }

}
