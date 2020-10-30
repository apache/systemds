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
import org.apache.sysds.runtime.privacy.PrivacyUtils;
import org.apache.sysds.runtime.privacy.finegrained.DataRange;
import org.apache.sysds.runtime.privacy.finegrained.FineGrainedPrivacy;

public abstract class AppendPropagator implements Propagator {

	protected MatrixBlock input1, input2;
	protected PrivacyConstraint privacyConstraint1, privacyConstraint2;

	public AppendPropagator(MatrixBlock input1, PrivacyConstraint privacyConstraint1,
		MatrixBlock input2, PrivacyConstraint privacyConstraint2){
		setFields(input1, privacyConstraint1, input2, privacyConstraint2);
	}

	public void setFields(MatrixBlock input1, PrivacyConstraint privacyConstraint1,
		MatrixBlock input2, PrivacyConstraint privacyConstraint2){
		this.input1 = input1;
		this.input2 = input2;
		this.privacyConstraint1 = privacyConstraint1;
		this.privacyConstraint2 = privacyConstraint2;
	}

	@Override
	public PrivacyConstraint propagate() {
		PrivacyConstraint mergedPrivacyConstraint = new PrivacyConstraint();
		propagateInput1Constraint(mergedPrivacyConstraint);
		propagateInput2Constraint(mergedPrivacyConstraint);
		return mergedPrivacyConstraint;
	}

	private void propagateInput1Constraint(PrivacyConstraint mergedPrivacyConstraint){
		if ( PrivacyUtils.privacyConstraintActivated(privacyConstraint1) ){
			//Get dimensions of input1 and make a fine-grained constraint of that size with privacy level
			long[] endDims = new long[]{input1.getNumRows()-1, input1.getNumColumns()-1};
			mergedPrivacyConstraint.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0}, endDims), privacyConstraint1.getPrivacyLevel());
		}
		if ( PrivacyUtils.privacyConstraintFineGrainedActivated(privacyConstraint1) ){
			privacyConstraint1.getFineGrainedPrivacy().getAllConstraintsList().forEach(
				constraint -> mergedPrivacyConstraint.getFineGrainedPrivacy().put(constraint.getKey(), constraint.getValue())
			);
		}
	}

	private void propagateInput2Constraint(PrivacyConstraint mergedPrivacyConstraint){
		if ( PrivacyUtils.privacyConstraintActivated(privacyConstraint2) ){
			//Get dimensions of input2 and ...
			long[] endDims = new long[]{input2.getNumRows()-1, input2.getNumColumns()-1};
			appendInput2(mergedPrivacyConstraint.getFineGrainedPrivacy(),
				new DataRange(new long[]{0,0}, endDims), privacyConstraint2.getPrivacyLevel());
		}
		if ( PrivacyUtils.privacyConstraintFineGrainedActivated(privacyConstraint2) ){
			// propagate fine-grained constraints
			privacyConstraint2.getFineGrainedPrivacy().getAllConstraintsList().forEach(
				constraint -> appendInput2(
					mergedPrivacyConstraint.getFineGrainedPrivacy(),
					constraint.getKey(), constraint.getValue()
				)
			);
		}
	}

	protected abstract void appendInput2(FineGrainedPrivacy mergedConstraint, DataRange range, PrivacyLevel privacyLevel);
}
