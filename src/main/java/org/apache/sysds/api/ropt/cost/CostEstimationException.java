package org.apache.sysds.api.ropt.cost;

/**
 * Exception thrown when the cost estimation gets in
 * a state that should not raise runtime a exception.
 * Such exception is to be raised only in the following case:
 * <li>Local memory is not sufficient for the estimated caching</li>
 */
public class CostEstimationException extends Exception {
    public CostEstimationException(String message) {
        super(message);
    }
}
