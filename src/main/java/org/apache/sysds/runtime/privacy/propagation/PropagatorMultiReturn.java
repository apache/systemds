package org.apache.sysds.runtime.privacy.propagation;

import org.apache.sysds.runtime.privacy.PrivacyConstraint;

/**
 * Interface for all propagator instances with multiple outputs.
 */
public interface PropagatorMultiReturn {
	/**
	 * Activates the propagation and returns the output privacy constraints.
	 * @return output privacy constraints.
	 */
	PrivacyConstraint[] propagate();
}
