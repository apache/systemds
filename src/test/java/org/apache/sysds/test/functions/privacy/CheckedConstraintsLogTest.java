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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.EnumMap;
import java.util.concurrent.atomic.LongAdder;

import org.apache.sysds.runtime.privacy.CheckedConstraintsLog;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Test;

@net.jcip.annotations.NotThreadSafe
public class CheckedConstraintsLogTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		CheckedConstraintsLog.reset();
	}

	@Test
	public void addCheckedConstraintsNull(){
		CheckedConstraintsLog.addCheckedConstraints(null);
		assertTrue(CheckedConstraintsLog.getCheckedConstraints() != null && CheckedConstraintsLog.getCheckedConstraints().isEmpty());
	}
	
	@Test
	public void addCheckedConstraintsEmpty(){
		EnumMap<PrivacyLevel,LongAdder> checked = new EnumMap<>(PrivacyLevel.class);
		CheckedConstraintsLog.addCheckedConstraints(checked);
		assertTrue(CheckedConstraintsLog.getCheckedConstraints() != null && CheckedConstraintsLog.getCheckedConstraints().isEmpty());
	}

	@Test
	public void addCheckedConstraintsSingleValue(){
		EnumMap<PrivacyLevel,LongAdder> checked = getMap(PrivacyLevel.Private, 300);
		CheckedConstraintsLog.addCheckedConstraints(checked);
		assertTrue(CheckedConstraintsLog.getCheckedConstraints().get(PrivacyLevel.Private).longValue() == 300);
	}

	@Test
	public void addCheckedConstraintsTwoValues(){
		EnumMap<PrivacyLevel,LongAdder> checked = getMap(PrivacyLevel.Private, 300);
		CheckedConstraintsLog.addCheckedConstraints(checked);
		EnumMap<PrivacyLevel,LongAdder> checked2 = getMap(PrivacyLevel.Private, 150);
		CheckedConstraintsLog.addCheckedConstraints(checked2);
		assertTrue(CheckedConstraintsLog.getCheckedConstraints().get(PrivacyLevel.Private).longValue() == 450);
	}

	@Test
	public void addCheckedConstraintsMultipleValues(){
		EnumMap<PrivacyLevel,LongAdder> checked = getMap(PrivacyLevel.Private, 300);
		CheckedConstraintsLog.addCheckedConstraints(checked);
		EnumMap<PrivacyLevel,LongAdder> checked2 = getMap(PrivacyLevel.Private, 150);
		CheckedConstraintsLog.addCheckedConstraints(checked2);
		EnumMap<PrivacyLevel,LongAdder> checked3 = getMap(PrivacyLevel.PrivateAggregation, 150);
		CheckedConstraintsLog.addCheckedConstraints(checked3);
		assertTrue(CheckedConstraintsLog.getCheckedConstraints().get(PrivacyLevel.Private).longValue() == 450 
		    && CheckedConstraintsLog.getCheckedConstraints().get(PrivacyLevel.PrivateAggregation).longValue() == 150);
	}

	private static EnumMap<PrivacyLevel,LongAdder> getMap(PrivacyLevel level, long value){
		EnumMap<PrivacyLevel,LongAdder> checked = new EnumMap<>(PrivacyLevel.class);
		LongAdder valueAdder = new LongAdder();
		valueAdder.add(value);
		checked.put(level, valueAdder);
		return checked;
	}

	@Test
	public void addLoadedConstraintsSingleValue(){
		Integer n = 12;
		for (int i = 0; i < n; i++)
			CheckedConstraintsLog.addLoadedConstraint(PrivacyLevel.Private);
		assertEquals(n.longValue(), CheckedConstraintsLog.getLoadedConstraints().get(PrivacyLevel.Private).longValue());
	}
}
