/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.privacy.propagation;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.runtime.privacy.finegrained.FineGrainedPrivacy;

/**
 * MatrixMultiplicationPropagator that overrides generateFineGrainedConstraints by finding the private elements first
 * followed by propagating PrivateAggregation in case of non-aggregating operator types.
 */
public class MatrixMultiplicationPropagatorPrivateFirst extends MatrixMultiplicationPropagator {

	public MatrixMultiplicationPropagatorPrivateFirst(){
		super();
	}

	public MatrixMultiplicationPropagatorPrivateFirst(MatrixBlock input1, PrivacyConstraint privacyConstraint1,
		MatrixBlock input2, PrivacyConstraint privacyConstraint2) {
		super(input1, privacyConstraint1, input2, privacyConstraint2);
	}

	/**
	 * Generates fine-grained constraints and puts them in the mergedFineGrainedConstraints instance.
	 * This implementation first loops over the rowPrivacy array and sets the entire row to private if the rowPrivacy is private.
	 * Next, it loops over the colPrivacy array and sets the entire column to private if the colPrivacy is private.
	 * Then it loops over the operatorTypes for both input matrices.
	 * If the operator type is non-aggregate and the privacy level is PrivateAggregation,
	 * then it sets the entire row/column to PrivateAggregation.
	 * If the operator type is non-aggregate and the privacy level is not private, then it loops through all elements
	 * in the row/column and checks for PrivateAggregation in the privacy level array of the other input.
	 */
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
				} else if ( colPrivacy[k] != PrivacyConstraint.PrivacyLevel.Private ){
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
