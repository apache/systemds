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

package org.apache.sysds.runtime.privacy;

import org.apache.sysds.runtime.privacy.FineGrained.DataRange;
import org.apache.sysds.runtime.privacy.FineGrained.FineGrainedPrivacy;
import org.apache.sysds.runtime.privacy.FineGrained.FineGrainedPrivacyList;

/**
 * PrivacyConstraint holds all privacy constraints for data in the system at
 * compile time and runtime.
 */
public class PrivacyConstraint
{
	public enum PrivacyLevel {
		None,               // No data exchange constraints. Data can be shared with anyone.
		Private,            // Data cannot leave the origin.
		PrivateAggregation  // Only aggregations of the data can leave the origin.
	}

	protected PrivacyLevel privacyLevel = PrivacyLevel.None;
	protected FineGrainedPrivacy fineGrainedPrivacy;
	
	/**
	 * Basic Constructor with a fine-grained collection 
	 * based on a list implementation.
	 */
	public PrivacyConstraint(){
		this(new FineGrainedPrivacyList());
	}

	/**
	 * Constructor with the option to choose between 
	 * different fine-grained collection implementations.
	 */
	public PrivacyConstraint(FineGrainedPrivacy fineGrainedPrivacyCollection){
		setFineGrainedPrivacyConstraints(fineGrainedPrivacyCollection);
	}

	/**
	 * Constructor with default fine-grained collection implementation
	 * where the entire data object is set to the given privacy level.
	 * @param privacyLevel for the entire data object.
	 */
	public PrivacyConstraint(PrivacyLevel privacyLevel) {
		this();
		setPrivacyLevel(privacyLevel);
	}

	public void setPrivacyLevel(PrivacyLevel privacyLevel){
		this.privacyLevel = privacyLevel;
	}

	public PrivacyLevel getPrivacyLevel(){
		return privacyLevel;
	}

	/**
	 * Checks if fine-grained privacy is set for this privacy constraint. 
	 * @return true if the privacy constraint has fine-grained constraints.
	 */
	public boolean hasFineGrainedConstraints(){
		return fineGrainedPrivacy.hasConstraints();
	}

	/**
	 * Sets fine-grained privacy for the privacy constraint. 
	 * Existing fine-grained privacy collection will be overwritten.
	 * @param fineGrainedPrivacy fine-grained privacy instance which is set for the privacy constraint
	 */
	public void setFineGrainedPrivacyConstraints(FineGrainedPrivacy fineGrainedPrivacy){
		this.fineGrainedPrivacy = fineGrainedPrivacy;
	}

	/**
	 * Get fine-grained privacy instance. 
	 * @return fine-grained privacy instance
	 */
	public FineGrainedPrivacy getFineGrainedPrivacy(){
		return fineGrainedPrivacy;
	}

	/**
	 * Return true if any of the elements has privacy level private
	 * @return true if any element has privacy level private
	 */
	public boolean hasPrivateElements(){
		if (privacyLevel == PrivacyLevel.Private) return true;
		if ( hasFineGrainedConstraints() ){
			DataRange[] dataRanges = fineGrainedPrivacy.getDataRangesOfPrivacyLevel(PrivacyLevel.Private);
			return dataRanges != null && dataRanges.length > 0;
		} else return false;
	}

	/**
	 * Return true if any constraints have level Private or PrivateAggregate.
	 * @return true if any constraints have level Private or PrivateAggregate
	 */
	public boolean hasConstraints(){
		if ( privacyLevel != null && 
			(privacyLevel == PrivacyLevel.Private || privacyLevel == PrivacyLevel.PrivateAggregation) )
			return true;
		else if ( hasFineGrainedConstraints() ){
			DataRange[] privateRanges = fineGrainedPrivacy.getDataRangesOfPrivacyLevel(PrivacyLevel.Private);
			DataRange[] aggregateRanges = fineGrainedPrivacy.getDataRangesOfPrivacyLevel(PrivacyLevel.PrivateAggregation);
			return (privateRanges != null && privateRanges.length > 0) 
				|| (aggregateRanges != null && aggregateRanges.length > 0);
		} else return false;
	}

}
