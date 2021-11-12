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

import java.util.EnumMap;
import java.util.concurrent.atomic.LongAdder;

import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;

public class PrivacyMonitor
{
	private static final EnumMap<PrivacyLevel,LongAdder> checkedConstraints;

	private static boolean checkPrivacy = false;

	static {
		checkedConstraints = new EnumMap<>(PrivacyLevel.class);
		for ( PrivacyLevel level : PrivacyLevel.values() ){
			checkedConstraints.put(level, new LongAdder());
		}
	}

	public static EnumMap<PrivacyLevel,LongAdder> getCheckedConstraints() {
		return checkedConstraints;
	}

	private static void incrementCheckedConstraints(PrivacyLevel privacyLevel) {
		if ( privacyLevel == null )
			throw new NullPointerException("Cannot increment checked constraints log: Privacy level is null.");
		checkedConstraints.get(privacyLevel).increment();
	}

	/**
	 * Update checked constraints log if checkPrivacy is activated.
	 * The checked constraints log is updated with both the general 
	 * privacy constraint and the fine-grained constraints.
	 * 
	 * @param privacyConstraint used for updating log
	 */
	private static void updateCheckedConstraintsLog(PrivacyConstraint privacyConstraint) {
		if ( checkPrivacy ){
			if ( privacyConstraint.privacyLevel != PrivacyLevel.None){
				incrementCheckedConstraints(privacyConstraint.privacyLevel);
			}
			if ( PrivacyUtils.privacyConstraintFineGrainedActivated(privacyConstraint) ){
				int privateNum = privacyConstraint.getFineGrainedPrivacy()
					.getDataRangesOfPrivacyLevel(PrivacyLevel.Private).length;
				int aggregateNum = privacyConstraint.getFineGrainedPrivacy()
					.getDataRangesOfPrivacyLevel(PrivacyLevel.PrivateAggregation).length;
				checkedConstraints.get(PrivacyLevel.Private).add(privateNum);
				checkedConstraints.get(PrivacyLevel.PrivateAggregation).add(aggregateNum);
			}
		}
	}

	/**
	 * Clears all checked constraints.
	 * This is used to reset the counter of checked constraints for each PrivacyLevel.
	 */
	public static void clearCheckedConstraints(){
		checkedConstraints.replaceAll((k,v)->new LongAdder());
	}

	public static void setCheckPrivacy(boolean checkPrivacyParam){
		checkPrivacy = checkPrivacyParam;
	}

	/**
	 * Throws DMLPrivacyException if privacy constraint is set to private or private aggregation.
	 * The checked constraints log will be updated before throwing an exception.
	 * @param dataObject input data object
	 * @return data object or data object with privacy constraint removed in case the privacy level was none.
	 */
	public static Data handlePrivacy(Data dataObject){
		if(dataObject == null)
			return null;
		PrivacyConstraint privacyConstraint = dataObject.getPrivacyConstraint();

		if ( PrivacyUtils.someConstraintSetUnary(privacyConstraint) ){
			updateCheckedConstraintsLog(privacyConstraint);
			throw new DMLPrivacyException("Cannot share variable, since the privacy constraint "
				+ "of the requested variable is activated");
		} else dataObject.setPrivacyConstraints(null);
		return dataObject;
	}
}
