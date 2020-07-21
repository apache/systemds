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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Map;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.PrivacyPropagator;
import org.apache.sysds.runtime.privacy.FineGrained.DataRange;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Test;

public class PrivacyPropagatorTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		// TODO Auto-generated method stub

	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneral(){
		MatrixBlock inputMatrix1 = new MatrixBlock(10,20,15);
		MatrixBlock inputMatrix2 = new MatrixBlock(20,30,12);
		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.Private);
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{3,8},new long[]{2,5}), PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		PrivacyConstraint mergedConstraint = PrivacyPropagator.matrixMultiplicationPropagation(inputMatrix1, constraint1, inputMatrix2, constraint2);
		assertTrue("Privacy should be set to Private", mergedConstraint.hasPrivateElements());
		assertFalse("Fine grained constraint should not be propagated", mergedConstraint.hasFineGrainedConstraints());
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateGeneral2(){
		MatrixBlock inputMatrix1 = new MatrixBlock(10,20,15);
		MatrixBlock inputMatrix2 = new MatrixBlock(20,30,12);
		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.Private);
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{3,8},new long[]{2,5}), PrivacyLevel.PrivateAggregation);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		PrivacyConstraint mergedConstraint = PrivacyPropagator.matrixMultiplicationPropagation(inputMatrix1, constraint1, inputMatrix2, constraint2);
		assertTrue("Privacy should be set to Private", mergedConstraint.hasPrivateElements());
		assertFalse("Fine grained constraint should not be propagated", mergedConstraint.hasFineGrainedConstraints());
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateFineGrained(){
		MatrixBlock inputMatrix1 = new MatrixBlock(4,3,2);
		MatrixBlock inputMatrix2 = new MatrixBlock(3,3,4);
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		PrivacyConstraint mergedConstraint = PrivacyPropagator.matrixMultiplicationPropagation(inputMatrix1, constraint1, inputMatrix2, constraint2);
		assertTrue("Privacy should be set to Private", mergedConstraint.hasPrivateElements());
		assertTrue("Fine grained constraint should not be propagated", mergedConstraint.hasFineGrainedConstraints());
		assertTrue("Merged constraint should not contain privacy level PrivateAggregation", mergedConstraint.getFineGrainedPrivacy().getDataRangesOfPrivacyLevel(PrivacyLevel.PrivateAggregation).length == 0);
		Map<DataRange, PrivacyLevel> outputElement1 = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevelOfElement(new long[]{1,0});
		Map<DataRange, PrivacyLevel> outputElement2 = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevelOfElement(new long[]{1,1});
		Map<DataRange, PrivacyLevel> outputElement3 = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevelOfElement(new long[]{1,2});
		assertEquals(1, outputElement1.size());
		assertEquals(1, outputElement2.size());
		assertEquals(1, outputElement3.size());
		assertTrue("Privacy level of element 1 is Private", outputElement1.containsValue(PrivacyLevel.Private));
		assertTrue("Privacy level of element 2 is Private", outputElement2.containsValue(PrivacyLevel.Private));
		assertTrue("Privacy level of element 3 is Private", outputElement3.containsValue(PrivacyLevel.Private));
		assertTrue("Any other index has no privacy constraint", mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(new DataRange(new long[]{2,0}, new long[]{3,2})).isEmpty() );
	}

	@Test
	public void matrixMultiplicationPropagationTestPrivateFineGrained2(){
		MatrixBlock inputMatrix1 = new MatrixBlock(4,3,2);
		MatrixBlock inputMatrix2 = new MatrixBlock(3,3,4);
		PrivacyConstraint constraint1 = new PrivacyConstraint();
		PrivacyConstraint constraint2 = new PrivacyConstraint();
		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
		PrivacyConstraint mergedConstraint = PrivacyPropagator.matrixMultiplicationPropagation(inputMatrix1, constraint1, inputMatrix2, constraint2);
		assertTrue("Privacy should be set to Private", mergedConstraint.hasPrivateElements());
		assertTrue("Fine grained constraint should not be propagated", mergedConstraint.hasFineGrainedConstraints());
		assertTrue("Merged constraint should not contain privacy level PrivateAggregation", mergedConstraint.getFineGrainedPrivacy().getDataRangesOfPrivacyLevel(PrivacyLevel.PrivateAggregation).length == 0);
		Map<DataRange, PrivacyLevel> outputRange = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(new DataRange(new long[]{0,0},new long[]{3,1}));
		assertEquals(8, outputRange.size());
		assertTrue("Privacy level is Private", outputRange.containsValue(PrivacyLevel.Private));
		assertTrue("Any other index has no privacy constraint", mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(new DataRange(new long[]{0,2}, new long[]{3,2})).isEmpty() );
	}
}