package org.apache.sysds.runtime.privacy.propagation;

import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.PrivacyUtils;
import org.apache.sysds.runtime.privacy.finegrained.DataRange;

import java.util.Map;

public class ListAppendPropagator implements Propagator {

	private final ListObject input1, input2;
	private final PrivacyConstraint privacyConstraint1, privacyConstraint2;

	public ListAppendPropagator(ListObject input1, PrivacyConstraint privacyConstraint1, ListObject input2, PrivacyConstraint privacyConstraint2){
		this.input1 = input1;
		this.input2 = input2;
		this.privacyConstraint1 = privacyConstraint1;
		this.privacyConstraint2 = privacyConstraint2;
	}

	@Override public PrivacyConstraint propagate() {
		PrivacyConstraint mergedPrivacyConstraint = new PrivacyConstraint();
		if(PrivacyUtils.privacyConstraintActivated(privacyConstraint1)) {
			mergedPrivacyConstraint.getFineGrainedPrivacy()
				.put(new DataRange(new long[] {0}, new long[] {input1.getLength()}), privacyConstraint1.getPrivacyLevel());
			privacyConstraint1.getFineGrainedPrivacy().getAllConstraintsList().forEach(
				constraint -> mergedPrivacyConstraint.getFineGrainedPrivacy().put(constraint.getKey(), constraint.getValue()));
		}
		if ( PrivacyUtils.privacyConstraintActivated(privacyConstraint2) ){
			mergedPrivacyConstraint.getFineGrainedPrivacy()
				.put(new DataRange(new long[]{input1.getLength()}, new long[]{input2.getLength()}), privacyConstraint2.getPrivacyLevel());
			if ( privacyConstraint2.hasFineGrainedConstraints() ){
				for ( Map.Entry<DataRange, PrivacyConstraint.PrivacyLevel> constraint : privacyConstraint2.getFineGrainedPrivacy().getAllConstraintsList()){
					long beginIndex = constraint.getKey().getBeginDims()[0] + input1.getLength();
					long endIndex = constraint.getKey().getEndDims()[0] + input1.getLength();
					mergedPrivacyConstraint.getFineGrainedPrivacy()
						.put(new DataRange(new long[]{beginIndex}, new long[]{endIndex}), constraint.getValue());
				}
			}
		}
		return mergedPrivacyConstraint;
	}
}
