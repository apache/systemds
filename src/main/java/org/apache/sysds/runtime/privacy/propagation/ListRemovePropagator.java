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
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.runtime.privacy.PrivacyUtils;
import org.apache.sysds.runtime.privacy.finegrained.DataRange;

import java.util.Map;

public class ListRemovePropagator implements PropagatorMultiReturn {
	private final ScalarObject removePosition;
	private final ListObject list;
	private final PrivacyConstraint listPrivacyConstraint;


	public ListRemovePropagator(ListObject list, PrivacyConstraint listPrivacyConstraint, ScalarObject removePosition){
		this.list = list;
		this.listPrivacyConstraint = listPrivacyConstraint;
		this.removePosition = removePosition;
	}

	@Override
	public PrivacyConstraint[] propagate() {
		//TODO: Also let output privacy depend on removePosition privacy constraint
		PrivacyConstraint output1PrivacyConstraint = new PrivacyConstraint();
		PrivacyConstraint output2PrivacyConstraint = new PrivacyConstraint();
		if ( PrivacyUtils.privacyConstraintActivated(listPrivacyConstraint) ){
			output1PrivacyConstraint.setPrivacyLevel(listPrivacyConstraint.getPrivacyLevel());
			output2PrivacyConstraint.setPrivacyLevel(listPrivacyConstraint.getPrivacyLevel());
		}
		if ( listPrivacyConstraint.hasFineGrainedConstraints() ){
			Map<DataRange, PrivacyLevel> output1Ranges = listPrivacyConstraint.getFineGrainedPrivacy()
				.getPrivacyLevel(new DataRange(new long[]{0}, new long[]{removePosition.getLongValue()}));
			output1Ranges.forEach(
				(range, privacyLevel) -> {
					long endDim = Long.min(range.getEndDims()[0], removePosition.getLongValue());
					DataRange cappedRange = new DataRange(range.getBeginDims(),new long[]{endDim});
					output1PrivacyConstraint.getFineGrainedPrivacy().put(cappedRange, privacyLevel);
				}
			);

			Map<DataRange, PrivacyLevel> output2Ranges = listPrivacyConstraint.getFineGrainedPrivacy()
				.getPrivacyLevel(new DataRange(new long[]{removePosition.getLongValue()+1}, new long[]{list.getLength()}));
			output2Ranges.forEach(
				(range, privacyLevel) -> {
					long shiftValue = removePosition.getLongValue() + 1;
					long[] beginDims = new long[]{range.getBeginDims()[0]-shiftValue};
					long[] endDims = new long[]{range.getEndDims()[0]-shiftValue};
					output2PrivacyConstraint.getFineGrainedPrivacy().put(new DataRange(beginDims, endDims), privacyLevel);
				}
			);
		}
		return new PrivacyConstraint[]{output1PrivacyConstraint, output2PrivacyConstraint};
	}
}
