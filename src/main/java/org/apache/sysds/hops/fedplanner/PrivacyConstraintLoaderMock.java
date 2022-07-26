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

package org.apache.sysds.hops.fedplanner;

import org.apache.sysds.api.DMLException;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;

/**
 * This class is a mockup of the PrivacyConstraintLoader which replaces the call to loadFederatedPrivacyConstraints.
 * This means that instead of loading the privacy constraints from the federated workers,
 * the constraint returned for each federated DataOp will have the privacy level specified in the constructor,
 * without sending any federated requests.
 */
public class PrivacyConstraintLoaderMock extends PrivacyConstraintLoader {

	private final PrivacyLevel privacyLevel;

	/**
	 * Creates a mock of PrivacyConstraintLoader where the
	 * given privacy level is given to all federated data.
	 * @param mockLevel string representing the privacy level used for the setting of privacy constraints
	 */
	public PrivacyConstraintLoaderMock(String mockLevel){
		try{
			this.privacyLevel = PrivacyLevel.valueOf(mockLevel);
		} catch(IllegalArgumentException ex){
			throw new DMLException("Privacy level loaded from config not recognized. Loaded from config: " + mockLevel, ex);
		}
	}

	/**
	 * Set privacy constraint of given hop to mocked privacy level.
	 * This mocks the behavior of the privacy constraint loader by
	 * setting the privacy constraint to a specific level for all
	 * federated data objects instead of retrieving the privacy constraints
	 * from the workers.
	 * @param hop for which privacy constraint is set
	 */
	@Override
	public void loadFederatedPrivacyConstraints(Hop hop){
		hop.setPrivacy(new PrivacyConstraint(privacyLevel));
	}
}
