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

package org.apache.sysds.runtime.privacy.finegrained;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;

/**
 * Simple implementation of retrieving fine-grained privacy constraints based on
 * iterating a LinkedHashMap.
 */
public class FineGrainedPrivacyMap implements FineGrainedPrivacy {

	private Map<DataRange, PrivacyLevel> constraintCollection = new LinkedHashMap<>();

	@Override
	public void put(DataRange dataRange, PrivacyLevel privacyLevel) {
		constraintCollection.put(dataRange, privacyLevel);
	}

	@Override
	public Map<DataRange, PrivacyLevel> getPrivacyLevel(DataRange searchRange) {
		Map<DataRange, PrivacyLevel> matches = new LinkedHashMap<>();
		constraintCollection.forEach((range, level) -> {
			if (range.overlaps(searchRange))
				matches.put(range, level);
		});
		return matches;
	}

	@Override
	public Map<DataRange, PrivacyLevel> getPrivacyLevelOfElement(long[] searchIndex) {
		Map<DataRange, PrivacyLevel> matches = new LinkedHashMap<>();
		constraintCollection.forEach((range, level) -> {
			if (range.contains(searchIndex))
				matches.put(range, level);
		});
		return matches;
	}

	@Override
	public DataRange[] getDataRangesOfPrivacyLevel(PrivacyLevel privacyLevel) {
		ArrayList<DataRange> matches = new ArrayList<>();
		constraintCollection.forEach((k, v) -> {
			if (v == privacyLevel)
				matches.add(k);
		});
		return matches.toArray(new DataRange[0]);
	}

	@Override
	public void removeAllConstraints() {
		constraintCollection.clear();
	}

	@Override
	public boolean hasConstraints() {
		return !constraintCollection.isEmpty();
	}

	@Override
	public Map<String, long[][][]> getAllConstraints() {
		ArrayList<long[][]> privateRanges = new ArrayList<>();
		ArrayList<long[][]> aggregateRanges = new ArrayList<>();
		constraintCollection.forEach((range, privacylevel) -> {
			if (privacylevel == PrivacyLevel.Private)
				privateRanges.add(new long[][] { range.getBeginDims(), range.getEndDims() });
			else if (privacylevel == PrivacyLevel.PrivateAggregation)
				aggregateRanges.add(new long[][] { range.getBeginDims(), range.getEndDims() });
		});
		Map<String, long[][][]> constraintMap = new LinkedHashMap<>();
		constraintMap.put(PrivacyLevel.Private.name(), privateRanges.toArray(new long[0][][]));
		constraintMap.put(PrivacyLevel.PrivateAggregation.name(), privateRanges.toArray(new long[0][][]));
		return constraintMap;
	}

	@Override
	public ArrayList<Entry<DataRange, PrivacyLevel>> getAllConstraintsList() {
		ArrayList<Map.Entry<DataRange, PrivacyLevel>> outputList = new ArrayList<>();
		constraintCollection.forEach((k,v)->outputList.add(new AbstractMap.SimpleEntry<>(k,v)));
		return outputList;
	}
}
