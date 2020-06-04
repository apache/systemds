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

/**
 * PrivacyConstraint holds all privacy constraints for data in the system at compile time and runtime. 
 */
public class PrivacyConstraint
{
	public enum PrivacyLevel {
		None,               // No data exchange constraints. Data can be shared with anyone.
		Private,            // Data cannot leave the origin.
		PrivateAggregation  // Only aggregations of the data can leave the origin.
	}

	protected PrivacyLevel privacyLevel = PrivacyLevel.None;

	public PrivacyConstraint(){}

	public PrivacyConstraint(PrivacyLevel privacyLevel) {
		setPrivacyLevel(privacyLevel);
	}

	public void setPrivacyLevel(PrivacyLevel privacyLevel){
		this.privacyLevel = privacyLevel;
	}

	public PrivacyLevel getPrivacyLevel(){
		return privacyLevel;
	}
}
