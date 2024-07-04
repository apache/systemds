package org.apache.sysds.resource.cost;

/**
 * Exception thrown when the cost estimation gets in
 * a state that should not raise runtime a exception.
 * Such exception is to be raised only in the following case:
 * <li>Local memory is not sufficient for the estimated caching</li>
 */
public class CostEstimationException extends Exception {
	private static final long serialVersionUID = -6709101762468084495L;

	public CostEstimationException(String message) {
		super(message);
	}
}
