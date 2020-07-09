package org.apache.sysds.runtime.privacy.FineGrained;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;

import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;

/**
 * Simple implementation of retrieving fine-grained privacy constraints
 * based on iterating a LinkedHashMap.
 */
public class FineGrainedPrivacyBrute implements FineGrainedPrivacy {

	private Map<DataRange, PrivacyLevel> constraintCollection = new LinkedHashMap<>();

	@Override
	public void put(DataRange dataRange, PrivacyLevel privacyLevel) {
		constraintCollection.put(dataRange, privacyLevel);
	}

	@Override
	public Map<DataRange, PrivacyLevel> getPrivacyLevel(DataRange searchRange) {
		Map<DataRange, PrivacyLevel> matches = new LinkedHashMap<>();
		constraintCollection.forEach((range,level) -> { if (range.overlaps(searchRange)) matches.put(range, level); } );
		return matches;
	}

	@Override
	public Map<DataRange, PrivacyLevel> getPrivacyLevelOfElement(long[] searchIndex) {
		Map<DataRange, PrivacyLevel> matches = new LinkedHashMap<>();
		constraintCollection.forEach((range,level) -> { if (range.contains(searchIndex)) matches.put(range, level); } );
		return matches;
	}

	@Override
	public DataRange[] getDataRangesOfPrivacyLevel(PrivacyLevel privacyLevel) {
		ArrayList<DataRange> matches = new ArrayList<>();
		constraintCollection.forEach((k,v) -> { if (v == privacyLevel) matches.add(k); } );
		return matches.toArray(new DataRange[0]);
	}

	@Override
	public void removeAllConstraints() {
		constraintCollection.clear();
	}
	
}