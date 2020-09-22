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
