package org.apache.sysds.runtime.compress.exceptions;

/**
 * Exception thrown when matrix compression fails.
 *
 * @author Nirvan C. UdaysinghJhurree
 */
public class CompressionException extends Exception {

    private static final long serialVersionUID = 1L;

    public CompressionException(String message) {
        super(message);
    }

    public CompressionException(String message, Throwable cause) {
        super(message, cause);
    }
}