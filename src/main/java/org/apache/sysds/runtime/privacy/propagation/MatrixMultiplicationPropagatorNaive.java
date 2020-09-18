package org.apache.sysds.runtime.privacy.propagation;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.runtime.privacy.PrivacyPropagator;
import org.apache.sysds.runtime.privacy.finegrained.FineGrainedPrivacy;

public class MatrixMultiplicationPropagatorNaive extends MatrixMultiplicationPropagator{

	public MatrixMultiplicationPropagatorNaive(){
		super();
	}

	public MatrixMultiplicationPropagatorNaive(MatrixBlock input1, PrivacyConstraint privacyConstraint1,
		MatrixBlock input2, PrivacyConstraint privacyConstraint2) {
		super(input1, privacyConstraint1, input2, privacyConstraint2);
	}

	@Override
	protected void generateFineGrainedConstraints(FineGrainedPrivacy mergedFineGrainedConstraints,
		PrivacyLevel[] rowPrivacy, PrivacyConstraint.PrivacyLevel[] colPrivacy,
		OperatorType[] operatorTypes1, OperatorType[] operatorTypes2) {
		for ( int i = 0; i < rowPrivacy.length; i++){
			for ( int j = 0; j < colPrivacy.length; j++){
				OperatorType operatorType = mergeOperatorType(operatorTypes1[i], operatorTypes2[j]);
				PrivacyLevel outputLevel = PrivacyPropagator.corePropagation(new PrivacyLevel[]{rowPrivacy[i], colPrivacy[j]}, operatorType);
				mergedFineGrainedConstraints.putElement(i,j,outputLevel);
			}
		}
	}
}
