package org.apache.sysds.runtime.privacy.FineGrained;

import java.util.Map;

import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;

public class FineGrainedPrivacyTree implements FineGrainedPrivacy {

	@Override
	public void put(DataRange dataRange, PrivacyLevel privacyLevel) {
		// TODO Auto-generated method stub

	}

	@Override
	public Map<DataRange, PrivacyLevel> getPrivacyLevel(DataRange searchRange) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Map<DataRange, PrivacyLevel> getPrivacyLevelOfElement(long[] searchIndex) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public DataRange[] getDataRangesOfPrivacyLevel(PrivacyLevel privacyLevel) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void removeAllConstraints() {
		// TODO Auto-generated method stub

	}
	
}