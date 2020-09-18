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

import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;
import org.apache.sysds.runtime.privacy.propagation.PrivacyPropagator;
import org.apache.sysds.runtime.privacy.propagation.OperatorType;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;

import static org.junit.Assert.assertEquals;

@RunWith(value = Parameterized.class)
public class CorePropagatorTest extends AutomatedTestBase {

	protected PrivacyLevel[] input;
	protected OperatorType operatorType;
	protected PrivacyLevel expectedPrivacyLevel;

	public CorePropagatorTest(PrivacyLevel[] input, OperatorType operatorType, PrivacyLevel expectedPrivacyLevel) {
		this.input = input;
		this.operatorType = operatorType;
		this.expectedPrivacyLevel = expectedPrivacyLevel;
	}

	@Override public void setUp() {}

	@Test
	public void corePropagation(){
		PrivacyLevel outputLevel = PrivacyPropagator.corePropagation(input, operatorType);
		assertEquals(expectedPrivacyLevel, outputLevel);
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][] {
			{new PrivacyLevel[]{PrivacyLevel.Private, PrivacyLevel.Private}, OperatorType.NonAggregate, PrivacyLevel.Private},
			{new PrivacyLevel[]{PrivacyLevel.Private, PrivacyLevel.Private}, OperatorType.Aggregate, PrivacyLevel.Private},
			{new PrivacyLevel[]{PrivacyLevel.Private, PrivacyLevel.PrivateAggregation}, OperatorType.NonAggregate, PrivacyLevel.Private},
			{new PrivacyLevel[]{PrivacyLevel.Private, PrivacyLevel.PrivateAggregation}, OperatorType.Aggregate, PrivacyLevel.Private},
			{new PrivacyLevel[]{PrivacyLevel.Private, PrivacyLevel.None}, OperatorType.NonAggregate, PrivacyLevel.Private},
			{new PrivacyLevel[]{PrivacyLevel.Private, PrivacyLevel.None}, OperatorType.Aggregate, PrivacyLevel.Private},
			{new PrivacyLevel[]{PrivacyLevel.PrivateAggregation, PrivacyLevel.Private}, OperatorType.NonAggregate, PrivacyLevel.Private},
			{new PrivacyLevel[]{PrivacyLevel.PrivateAggregation, PrivacyLevel.Private}, OperatorType.Aggregate, PrivacyLevel.Private},
			{new PrivacyLevel[]{PrivacyLevel.None, PrivacyLevel.Private}, OperatorType.NonAggregate, PrivacyLevel.Private},
			{new PrivacyLevel[]{PrivacyLevel.None, PrivacyLevel.Private}, OperatorType.Aggregate, PrivacyLevel.Private},
			{new PrivacyLevel[]{PrivacyLevel.PrivateAggregation, PrivacyLevel.PrivateAggregation}, OperatorType.NonAggregate, PrivacyLevel.PrivateAggregation},
			{new PrivacyLevel[]{PrivacyLevel.PrivateAggregation, PrivacyLevel.PrivateAggregation}, OperatorType.Aggregate, PrivacyLevel.None},
			{new PrivacyLevel[]{PrivacyLevel.None, PrivacyLevel.None}, OperatorType.NonAggregate, PrivacyLevel.None},
			{new PrivacyLevel[]{PrivacyLevel.None, PrivacyLevel.None}, OperatorType.Aggregate, PrivacyLevel.None},
			{new PrivacyLevel[]{PrivacyLevel.PrivateAggregation, PrivacyLevel.None}, OperatorType.NonAggregate, PrivacyLevel.PrivateAggregation},
			{new PrivacyLevel[]{PrivacyLevel.PrivateAggregation, PrivacyLevel.None}, OperatorType.Aggregate, PrivacyLevel.None},
			{new PrivacyLevel[]{PrivacyLevel.None, PrivacyLevel.PrivateAggregation}, OperatorType.NonAggregate, PrivacyLevel.PrivateAggregation},
			{new PrivacyLevel[]{PrivacyLevel.None, PrivacyLevel.PrivateAggregation}, OperatorType.Aggregate, PrivacyLevel.None}
		};
		return Arrays.asList(data);
	}
}
