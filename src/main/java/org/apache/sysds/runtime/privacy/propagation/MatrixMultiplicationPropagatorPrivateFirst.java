package org.apache.sysds.runtime.privacy.propagation;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.runtime.privacy.finegrained.FineGrainedPrivacy;

public class MatrixMultiplicationPropagatorPrivateFirst extends MatrixMultiplicationPropagator {

	public MatrixMultiplicationPropagatorPrivateFirst(){
		super();
	}

	public MatrixMultiplicationPropagatorPrivateFirst(MatrixBlock input1, PrivacyConstraint privacyConstraint1,
		MatrixBlock input2, PrivacyConstraint privacyConstraint2) {
		super(input1, privacyConstraint1, input2, privacyConstraint2);
	}

	@Override
	protected void generateFineGrainedConstraints(FineGrainedPrivacy mergedFineGrainedConstraints,
		PrivacyLevel[] rowPrivacy, PrivacyLevel[] colPrivacy,
		OperatorType[] operatorTypes1, OperatorType[] operatorTypes2) {
		int r1 = rowPrivacy.length;
		int c2 = colPrivacy.length;
		for ( int i = 0; i < rowPrivacy.length; i++ ) {
			if(rowPrivacy[i] == PrivacyConstraint.PrivacyLevel.Private) {
				// mark entire row private
				mergedFineGrainedConstraints.putRow(i,c2, PrivacyConstraint.PrivacyLevel.Private);
			}
		}

		for ( int j = 0; j < colPrivacy.length; j++ ) {
			if(colPrivacy[j] == PrivacyConstraint.PrivacyLevel.Private) {
				// mark entire col private
				mergedFineGrainedConstraints.putCol(j,r1, PrivacyConstraint.PrivacyLevel.Private);
			}
		}

		for ( int k = 0; k < operatorTypes1.length; k++ ){
			if ( operatorTypes1[k] == OperatorType.NonAggregate ){
				if ( rowPrivacy[k] == PrivacyConstraint.PrivacyLevel.PrivateAggregation ){
					// Mark entire row PrivateAggregation
					mergedFineGrainedConstraints.putRow(k,c2, PrivacyConstraint.PrivacyLevel.PrivateAggregation);
				} else if ( rowPrivacy[k] != PrivacyConstraint.PrivacyLevel.Private ){
					// Go through each element of colPrivacy and if element is PrivateAggregation then mark cell as PrivateAggregation
					for ( int l = 0; l < colPrivacy.length; l++ ){
						if ( colPrivacy[l] == PrivacyConstraint.PrivacyLevel.PrivateAggregation )
							mergedFineGrainedConstraints.putElement(k,l, PrivacyConstraint.PrivacyLevel.PrivateAggregation);
					}
				}
			}
		}

		// Do the same for operatorTypes2
		for ( int k = 0; k < operatorTypes2.length; k++ ){
			if ( operatorTypes2[k] == OperatorType.NonAggregate ){
				if ( colPrivacy[k] == PrivacyConstraint.PrivacyLevel.PrivateAggregation ){
					// Mark entire col PrivateAggregation
					mergedFineGrainedConstraints.putCol(k,r1, PrivacyConstraint.PrivacyLevel.PrivateAggregation);
				} else if ( rowPrivacy[k] != PrivacyConstraint.PrivacyLevel.Private ){
					// Go through each element of rowPrivacy and if element is PrivateAggregation then mark cell as PrivateAggregation
					for ( int l = 0; l < rowPrivacy.length; l++ ){
						if ( rowPrivacy[l] == PrivacyConstraint.PrivacyLevel.PrivateAggregation )
							mergedFineGrainedConstraints.putElement(k,l, PrivacyConstraint.PrivacyLevel.PrivateAggregation);
					}
				}
			}
		}
	}
}
