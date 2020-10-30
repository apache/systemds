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
	private final PrivacyConstraint removePositionPrivacyConstraint;
	private final ListObject list;
	private final PrivacyConstraint listPrivacyConstraint;


	public ListRemovePropagator(ListObject list, PrivacyConstraint listPrivacyConstraint, ScalarObject removePosition, PrivacyConstraint removePositionPrivacyConstraint){
		this.list = list;
		this.listPrivacyConstraint = listPrivacyConstraint;
		this.removePosition = removePosition;
		this.removePositionPrivacyConstraint = removePositionPrivacyConstraint;
	}

	@Override
	public PrivacyConstraint[] propagate() {
		PrivacyConstraint output1PrivacyConstraint = new PrivacyConstraint();
		PrivacyConstraint output2PrivacyConstraint = new PrivacyConstraint();
		propagateGeneralConstraints(output1PrivacyConstraint, output2PrivacyConstraint);
		propagateFineGrainedConstraints(output1PrivacyConstraint, output2PrivacyConstraint);
		return new PrivacyConstraint[]{output1PrivacyConstraint, output2PrivacyConstraint};
	}

	private void propagateFineGrainedConstraints(PrivacyConstraint output1PrivacyConstraint, PrivacyConstraint output2PrivacyConstraint){
		if ( PrivacyUtils.privacyConstraintFineGrainedActivated(listPrivacyConstraint) ){
			propagateFirstHalf(output1PrivacyConstraint);
			propagateSecondHalf(output1PrivacyConstraint);
			propagateRemovedElement(output2PrivacyConstraint);
		}
	}

	private void propagateFirstHalf(PrivacyConstraint output1PrivacyConstraint){
		// The newEndDimension is minus 2 since removePosition is given in 1-index terms whereas the data
		// and privacy constraints are 0-index and the privacy constraints are given as closed intervals
		long[] newEndDimension = new long[]{removePosition.getLongValue()-2};
		Map<DataRange, PrivacyLevel> output1Ranges = listPrivacyConstraint.getFineGrainedPrivacy()
			.getPrivacyLevel(new DataRange(new long[]{0}, newEndDimension));
		output1Ranges.forEach(
			(range, privacyLevel) -> {
				long endDim = Long.min(range.getEndDims()[0], removePosition.getLongValue()-2);
				DataRange cappedRange = new DataRange(range.getBeginDims(),new long[]{endDim});
				output1PrivacyConstraint.getFineGrainedPrivacy().put(cappedRange, privacyLevel);
			}
		);
	}

	private void propagateSecondHalf(PrivacyConstraint output1PrivacyConstraint){
		Map<DataRange, PrivacyLevel> output2Ranges = listPrivacyConstraint.getFineGrainedPrivacy()
			.getPrivacyLevel(new DataRange(new long[]{removePosition.getLongValue()}, new long[]{list.getLength()}));
		output2Ranges.forEach(
			(range, privacyLevel) -> {
				long[] beginDims = new long[]{range.getBeginDims()[0]-1};
				long[] endDims = new long[]{range.getEndDims()[0]-1};
				output1PrivacyConstraint.getFineGrainedPrivacy().put(new DataRange(beginDims, endDims), privacyLevel);
			}
		);
	}

	private void propagateRemovedElement(PrivacyConstraint output2PrivacyConstraint){
		if ( output2PrivacyConstraint.getPrivacyLevel() != PrivacyLevel.Private ){
			Map<DataRange, PrivacyLevel> elementPrivacy = listPrivacyConstraint.getFineGrainedPrivacy()
				.getPrivacyLevelOfElement(new long[]{removePosition.getLongValue()-1});
			if ( elementPrivacy.containsValue(PrivacyLevel.Private) )
				output2PrivacyConstraint.setPrivacyLevel(PrivacyLevel.Private);
			else if ( elementPrivacy.containsValue(PrivacyLevel.PrivateAggregation) )
				output2PrivacyConstraint.setPrivacyLevel(PrivacyLevel.PrivateAggregation);
		}
	}

	private void propagateGeneralConstraints(PrivacyConstraint output1PrivacyConstraint, PrivacyConstraint output2PrivacyConstraint){
		PrivacyLevel[] inputPrivacyLevels = PrivacyUtils.getGeneralPrivacyLevels(new PrivacyConstraint[]{
			listPrivacyConstraint, removePositionPrivacyConstraint
		});
		PrivacyLevel outputPrivacyLevel = PrivacyPropagator.corePropagation(inputPrivacyLevels, OperatorType.NonAggregate);
		output1PrivacyConstraint.setPrivacyLevel(outputPrivacyLevel);
		output2PrivacyConstraint.setPrivacyLevel(outputPrivacyLevel);
	}
}
