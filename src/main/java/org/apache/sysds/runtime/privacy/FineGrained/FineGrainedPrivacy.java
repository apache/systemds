package org.apache.sysds.runtime.privacy.FineGrained;

import java.util.Map;

import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;

public interface FineGrainedPrivacy {

	/**
	 * Set privacy level of the given data range.
	 * @param dataRange representing the range for which the privacy is set
	 * @param privacyLevel the level of privacy for the given data range
	 */
	public void put(DataRange dataRange, PrivacyLevel privacyLevel);

	/**
	 * Get the data ranges and related privacy levels within given data search range.
	 * @param searchRange the range from which all privacy levels are retrieved
	 * @return all mappings from range to privacy level within the given search range
	 */
	public Map<DataRange,PrivacyLevel> getPrivacyLevel(DataRange searchRange);

	/**
	 * Get the data ranges and related privacy levels of the element with the given index.
	 * @param searchIndex index of element
	 * @return all mappings from range to privacy level for the given search element
	 */
	public Map<DataRange,PrivacyLevel> getPrivacyLevelOfElement(long[] searchIndex);

	/**
	 * Get all data ranges for the given privacy level.
	 * @param privacyLevel for which data ranges are found
	 * @return all data ranges with the given privacy level
	 */
	public DataRange[] getDataRangesOfPrivacyLevel(PrivacyLevel privacyLevel);

	/**
	 * Remove all fine-grained privacy constraints.
	 */
	public void removeAllConstraints();
	
}