package org.apache.sysds.runtime.privacy.propagation;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.finegrained.DataRange;
import org.apache.sysds.runtime.privacy.finegrained.FineGrainedPrivacy;

public class CBindPropagator extends AppendPropagator {

	public CBindPropagator(MatrixBlock input1, PrivacyConstraint privacyConstraint1, MatrixBlock input2,
		PrivacyConstraint privacyConstraint2) {
		super(input1, privacyConstraint1, input2, privacyConstraint2);
	}

	@Override
	protected void appendInput1(FineGrainedPrivacy mergedConstraint, DataRange range,
		PrivacyConstraint.PrivacyLevel privacyLevel) {
		mergedConstraint.put(range, privacyLevel);
	}

	@Override
	protected void appendInput2(FineGrainedPrivacy mergedConstraint, DataRange range,
		PrivacyConstraint.PrivacyLevel privacyLevel) {
		long rowBegin = range.getBeginDims()[0]; //same as before
		long colBegin = range.getBeginDims()[1] + input1.getNumColumns();
		long[] beginDims = new long[]{rowBegin, colBegin};

		long rowEnd = range.getEndDims()[0]; //same as before
		long colEnd = range.getEndDims()[1] + input1.getNumColumns();
		long[] endDims = new long[]{rowEnd, colEnd};

		DataRange outputRange = new DataRange(beginDims, endDims);

		mergedConstraint.put(outputRange, privacyLevel);
	}
}
