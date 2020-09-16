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

import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.finegrained.FineGrainedPrivacy;

/**
 * MatrixMultiplicationPropagator that overrides generateFineGrainedConstraints by finding the private elements first
 * while propagating PrivateAggregation in case of non-aggregating operator types. This is an optimized version of
 * MatrixMultiplicationPropagatorPrivateFirst since it does both kinds of propagation in the same loop.
 */
public class MatrixMultiplicationPropagatorPrivateFirstOptimized extends MatrixMultiplicationPropagator {

	/**
	 * Generates fine-grained constraints and puts them in the mergedFineGrainedConstraints instance.
	 * This implementation loops over the rowPrivacy array and sets the entire row to private if the rowPrivacy is private.
	 * During this loop, if the element is not private, it will check the operator type and check for private aggregation level.
	 * If the operator type is non-aggregate and the privacy level is PrivateAggregation,
	 * then it sets the entire row/column to PrivateAggregation.
	 * If the operator type is non-aggregate and the privacy level is not private, then it loops through all elements
	 * in the row/column and checks for PrivateAggregation in the privacy level array of the other input.
	 */
	@Override
	protected void generateFineGrainedConstraints(FineGrainedPrivacy mergedFineGrainedConstraints,
		PrivacyConstraint.PrivacyLevel[] rowPrivacy, PrivacyConstraint.PrivacyLevel[] colPrivacy,
		OperatorType[] operatorTypes1, OperatorType[] operatorTypes2) {
		int r1 = rowPrivacy.length;
		int c2 = colPrivacy.length;
		for ( int i = 0; i < rowPrivacy.length; i++ ) {
			if(rowPrivacy[i] == PrivacyConstraint.PrivacyLevel.Private) {
				// mark entire row private
				mergedFineGrainedConstraints.putRow(i,c2, PrivacyConstraint.PrivacyLevel.Private);
			}
			else if ( operatorTypes1[i] == OperatorType.NonAggregate ){
				if ( rowPrivacy[i] == PrivacyConstraint.PrivacyLevel.PrivateAggregation ){
					// Mark entire row PrivateAggregation
					mergedFineGrainedConstraints.putRow(i,c2, PrivacyConstraint.PrivacyLevel.PrivateAggregation);
				} else { // rowPrivacy[i] is None, but colPrivacy could be PrivateAggregation
					// Go through each element of colPrivacy and if element is PrivateAggregation then mark cell as PrivateAggregation
					for ( int l = 0; l < colPrivacy.length; l++ ){
						if ( colPrivacy[l] == PrivacyConstraint.PrivacyLevel.PrivateAggregation )
							mergedFineGrainedConstraints.putElement(i,l, PrivacyConstraint.PrivacyLevel.PrivateAggregation);
					}
				}
			}
		}

		for ( int j = 0; j < colPrivacy.length; j++ ) {
			if(colPrivacy[j] == PrivacyConstraint.PrivacyLevel.Private) {
				// mark entire col private
				mergedFineGrainedConstraints.putCol(j,r1, PrivacyConstraint.PrivacyLevel.Private);
			}
			else if ( operatorTypes2[j] == OperatorType.NonAggregate ){
				if ( colPrivacy[j] == PrivacyConstraint.PrivacyLevel.PrivateAggregation ){
					// Mark entire col PrivateAggregation
					mergedFineGrainedConstraints.putCol(j,r1, PrivacyConstraint.PrivacyLevel.PrivateAggregation);
				} else { // colPrivacy[j] is None, but rowPrivacy could be PrivateAggregation
					// Go through each element of rowPrivacy and if element is PrivateAggregation then mark cell as PrivateAggregation
					for ( int l = 0; l < rowPrivacy.length; l++ ){
						if ( rowPrivacy[l] == PrivacyConstraint.PrivacyLevel.PrivateAggregation )
							mergedFineGrainedConstraints.putElement(j,l, PrivacyConstraint.PrivacyLevel.PrivateAggregation);
					}
				}
			}
		}
	}
}
