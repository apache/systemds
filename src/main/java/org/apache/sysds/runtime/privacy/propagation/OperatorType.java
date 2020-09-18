package org.apache.sysds.runtime.privacy.propagation;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.privacy.PrivacyPropagator;

public enum OperatorType {
	Aggregate,
	NonAggregate;
}
