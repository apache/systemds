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
 * MatrixMultiplicationPropagator that overrides generateFineGrainedConstraints
 * with a naive propagation of the fine-grained constraints.
 * The output is correct, but is likely less efficient than other implementations.
 */
public class MatrixMultiplicationPropagatorNaive extends MatrixMultiplicationPropagator{

	public MatrixMultiplicationPropagatorNaive(){
		super();
	}

	public MatrixMultiplicationPropagatorNaive(MatrixBlock input1, PrivacyConstraint privacyConstraint1,
		MatrixBlock input2, PrivacyConstraint privacyConstraint2) {
		super(input1, privacyConstraint1, input2, privacyConstraint2);
	}

	/**
	 * Generates fine-grained constraints and puts them in the mergedFineGrainedConstraints instance.
	 * This implementation loops over every cell of the output matrix and sets the constraint based on the
	 * row/column privacy constraints and the row/column operator type.
	 */
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
