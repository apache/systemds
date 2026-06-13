package org.apache.sysds.runtime.compress.exceptions;

/**
 * Exception thrown when matrix decompression fails.
 *
 * @author Nirvan C. Udaysingh Jhurree
 */
public class DecompressionException extends Exception {

    private static final long serialVersionUID = 1L;

    public DecompressionException(String message) {
        super(message);
    }

    public DecompressionException(String message, Throwable cause) {
        super(message, cause);
    }
}