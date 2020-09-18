package org.apache.sysds.runtime.privacy.propagation;

import org.apache.sysds.runtime.privacy.PrivacyConstraint;

public interface Propagator {
	PrivacyConstraint propagate();
}
