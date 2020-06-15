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

package org.apache.sysds.test.functions.privacy;

import java.util.HashMap;
import java.util.concurrent.atomic.LongAdder;

import org.apache.sysds.runtime.privacy.CheckedConstraintsLog;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Test;

@net.jcip.annotations.NotThreadSafe
public class CheckedConstraintsLogTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		CheckedConstraintsLog.getCheckedConstraints().clear();
	}

	@Test
	public void addCheckedConstraintsNull(){
		CheckedConstraintsLog.addCheckedConstraints(null);
		assert(CheckedConstraintsLog.getCheckedConstraints() != null && CheckedConstraintsLog.getCheckedConstraints().isEmpty());
	}
	
	@Test
	public void addCheckedConstraintsEmpty(){
		HashMap<PrivacyLevel,LongAdder> checked = new HashMap<>();
		CheckedConstraintsLog.addCheckedConstraints(checked);
		assert(CheckedConstraintsLog.getCheckedConstraints() != null && CheckedConstraintsLog.getCheckedConstraints().isEmpty());
	}

	@Test
	public void addCheckedConstraintsSingleValue(){
		HashMap<PrivacyLevel,LongAdder> checked = getMap(PrivacyLevel.Private, 300);
		CheckedConstraintsLog.addCheckedConstraints(checked);
		assert(CheckedConstraintsLog.getCheckedConstraints().get(PrivacyLevel.Private).longValue() == 300);
	}

	@Test
	public void addCheckedConstraintsTwoValues(){
		HashMap<PrivacyLevel,LongAdder> checked = getMap(PrivacyLevel.Private, 300);
		CheckedConstraintsLog.addCheckedConstraints(checked);
		HashMap<PrivacyLevel,LongAdder> checked2 = getMap(PrivacyLevel.Private, 150);
		CheckedConstraintsLog.addCheckedConstraints(checked2);
		assert(CheckedConstraintsLog.getCheckedConstraints().get(PrivacyLevel.Private).longValue() == 450);
	}

	@Test
	public void addCheckedConstraintsMultipleValues(){
		HashMap<PrivacyLevel,LongAdder> checked = getMap(PrivacyLevel.Private, 300);
		CheckedConstraintsLog.addCheckedConstraints(checked);
		HashMap<PrivacyLevel,LongAdder> checked2 = getMap(PrivacyLevel.Private, 150);
		CheckedConstraintsLog.addCheckedConstraints(checked2);
		HashMap<PrivacyLevel,LongAdder> checked3 = getMap(PrivacyLevel.PrivateAggregation, 150);
		CheckedConstraintsLog.addCheckedConstraints(checked3);
		assert(CheckedConstraintsLog.getCheckedConstraints().get(PrivacyLevel.Private).longValue() == 450 
		    && CheckedConstraintsLog.getCheckedConstraints().get(PrivacyLevel.PrivateAggregation).longValue() == 150);
	}

	private HashMap<PrivacyLevel,LongAdder> getMap(PrivacyLevel level, long value){
		HashMap<PrivacyLevel,LongAdder> checked = new HashMap<>();
		LongAdder valueAdder = new LongAdder();
		valueAdder.add(value);
		checked.put(level, valueAdder);
		return checked;
	}
}