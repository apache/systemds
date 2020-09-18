package org.apache.sysds.runtime.privacy.propagation;

import org.apache.sysds.runtime.privacy.PrivacyConstraint;

/**
 * Interface for all propagator instances.
 */
public interface Propagator {
	/**
	 * Activates the propagation and returns the output privacy constraint.
	 * @return output privacy constraint.
	 */
	PrivacyConstraint propagate();
}
