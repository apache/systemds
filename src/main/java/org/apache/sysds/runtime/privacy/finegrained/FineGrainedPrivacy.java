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

import java.util.ArrayList;
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

	/**
	 * True if any fine-grained constraints has been set. 
	 * @return true if any fine-grained constraint is set
	 */
	public boolean hasConstraints();

	/**
	 * Get all fine-grained constraints as a map from privacy level to 
	 * an array of data ranges represented as two-dimensional long arrays.
	 * @return map from privacy level to array of data ranges
	 */
	public Map<String, long[][][]> getAllConstraints();

	/**
	 * Return all fine-grained privacy constraints as an arraylist. 
	 * @return all constraints
	 */
	public ArrayList<Map.Entry<DataRange, PrivacyLevel>> getAllConstraintsList();
}
