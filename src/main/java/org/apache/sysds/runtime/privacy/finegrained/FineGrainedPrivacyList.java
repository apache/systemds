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
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;

/**
 * Simple implementation of retrieving fine-grained privacy constraints
 * based on pairs in an ArrayList.
 */
public class FineGrainedPrivacyList implements FineGrainedPrivacy {

	private final ArrayList<Map.Entry<DataRange, PrivacyLevel>> constraintCollection = new ArrayList<>();

	@Override
	public PrivacyLevel[] getRowPrivacy(int numRows, int numCols) {
		PrivacyLevel[] privacyLevels = new PrivacyLevel[numRows];
		for (int i = 0; i < numRows; i++) {
			long[] beginRange1 = new long[] {i, 0};
			long[] endRange1 = new long[] {i, numCols};
			privacyLevels[i] = getStrictestPrivacyLevel(new DataRange(beginRange1, endRange1));
		}
		return privacyLevels;
	}

	@Override
	public PrivacyLevel[] getColPrivacy(int numRows, int numCols) {
		PrivacyLevel[] privacyLevels = new PrivacyLevel[numCols];
		for (int i = 0; i < numCols; i++) {
			long[] beginRange = new long[] {0, i};
			long[] endRange = new long[] {numRows, i};
			privacyLevels[i] = getStrictestPrivacyLevel(new DataRange(beginRange, endRange));
		}
		return privacyLevels;
	}

	private PrivacyLevel getStrictestPrivacyLevel(DataRange searchRange){
		PrivacyLevel strictestLevel = PrivacyLevel.None;
		for ( Map.Entry<DataRange, PrivacyLevel> constraint : constraintCollection ) {
			if(constraint.getKey().overlaps(searchRange)) {
				if(constraint.getValue() == PrivacyLevel.Private)
					return PrivacyLevel.Private;
				if(constraint.getValue() == PrivacyLevel.PrivateAggregation)
					strictestLevel = PrivacyLevel.PrivateAggregation;
			}
		}
		return strictestLevel;
	}

	@Override
	public void put(DataRange dataRange, PrivacyLevel privacyLevel) {
		constraintCollection.add(new AbstractMap.SimpleEntry<>(dataRange, privacyLevel));
	}

	@Override
	public void putRow(int rowIndex, int rowLength, PrivacyLevel privacyLevel){
		put(new DataRange(new long[] {rowIndex, 0}, new long[] {rowIndex, rowLength-1}), privacyLevel);
	}

	@Override
	public void putCol(int colIndex, int colLength, PrivacyLevel privacyLevel){
		put(new DataRange(new long[] {0, colIndex}, new long[] {colLength-1, colIndex}), privacyLevel);
	}

	@Override
	public void putElement(int rowIndex, int colIndex, PrivacyLevel privacyLevel){
		put(new DataRange(new long[]{rowIndex,colIndex},new long[]{rowIndex,colIndex}), privacyLevel);
	}

	@Override
	public Map<DataRange,PrivacyLevel> getPrivacyLevel(DataRange searchRange) {
		Map<DataRange, PrivacyLevel> matches = new LinkedHashMap<>();
		for ( Map.Entry<DataRange, PrivacyLevel> constraint : constraintCollection ){
			if ( constraint.getKey().overlaps(searchRange) ) 
				matches.put(constraint.getKey(), constraint.getValue());
		}
		return matches;
	}

	@Override
	public Map<DataRange,PrivacyLevel> getPrivacyLevelOfElement(long[] searchIndex) {
		Map<DataRange, PrivacyLevel> matches = new LinkedHashMap<>();
		constraintCollection.forEach( constraint -> { 
			if (constraint.getKey().contains(searchIndex)) 
				matches.put(constraint.getKey(), constraint.getValue()); 
		} );
		return matches;
	}

	@Override
	public DataRange[] getDataRangesOfPrivacyLevel(PrivacyLevel privacyLevel) {
		ArrayList<DataRange> matches = new ArrayList<>();
		constraintCollection.forEach(constraint -> {
			if (constraint.getValue() == privacyLevel) 
				matches.add(constraint.getKey());
		} );
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
		constraintCollection.forEach(constraint -> {
			if ( constraint.getValue() == PrivacyLevel.Private )
				privateRanges.add(new long[][]{constraint.getKey().getBeginDims(), constraint.getKey().getEndDims()});
			else if ( constraint.getValue() == PrivacyLevel.PrivateAggregation )
				aggregateRanges.add(new long[][]{constraint.getKey().getBeginDims(), constraint.getKey().getEndDims()});
		});
		Map<String, long[][][]> constraintMap = new HashMap<>();
		constraintMap.put(PrivacyLevel.Private.name(), privateRanges.toArray(new long[0][][]));
		constraintMap.put(PrivacyLevel.PrivateAggregation.name(), aggregateRanges.toArray(new long[0][][]));
		return constraintMap;
	}

	@Override
	public ArrayList<Map.Entry<DataRange, PrivacyLevel>> getAllConstraintsList() {
		return constraintCollection;
	}

	@Override
	public boolean equals(Object other){
		if ( other instanceof FineGrainedPrivacyList ){
			FineGrainedPrivacyList otherFGP = (FineGrainedPrivacyList) other;
			if ( !otherFGP.hasConstraints() && !hasConstraints() )
				return true;
			if ( !otherFGP.hasConstraints() || !hasConstraints() )
				return false;
			return listEquals(otherFGP.getAllConstraintsList());
		}
		return false;
	}

	private boolean listEquals(ArrayList<Map.Entry<DataRange,PrivacyLevel>> otherFGP){
		if ( otherFGP.size() != constraintCollection.size() )
			return false;
		for ( Map.Entry<DataRange, PrivacyLevel> constraint : constraintCollection){
			if ( !otherFGP.contains(constraint) )
				return false;
		}
		return true;
	}

	@Override
	public String toString(){
		StringBuilder stringBuilder = new StringBuilder();
		for ( Map.Entry<DataRange,PrivacyLevel> entry : constraintCollection )
			stringBuilder.append(String.format("%s : %s",entry.getKey().toString(), entry.getValue().name()));
		return stringBuilder.toString();
	}
}
