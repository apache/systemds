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

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.runtime.privacy.finegrained.DataRange;
import org.apache.sysds.runtime.privacy.finegrained.FineGrainedPrivacy;
import org.apache.sysds.runtime.privacy.finegrained.FineGrainedPrivacyList;
import org.apache.sysds.runtime.privacy.finegrained.FineGrainedPrivacyMap;
import org.junit.After;
import org.junit.Test;
import org.junit.runners.Parameterized;
import org.junit.runner.RunWith;

@RunWith(Parameterized.class)
public class FineGrainedPrivacyTest {

	private FineGrainedPrivacy constraints;

	public FineGrainedPrivacyTest(FineGrainedPrivacy constraints){
		this.constraints = constraints;
	}

	@Parameterized.Parameters
	public static Collection<FineGrainedPrivacy[]> FineGrainedPrivacy(){
		return Arrays.asList(new FineGrainedPrivacy[][] {
			{new FineGrainedPrivacyMap()},
			{new FineGrainedPrivacyList()}
		});
	}

	@After
	public void setConstraintsToNull(){
		constraints.removeAllConstraints();
		constraints = null;
	}

	@Test
	public void getPrivacyLevelSingleConstraintCompletelyInRangeTest(){
		DataRange inputDataRange = new DataRange(new long[]{3L,2L,7L}, new long[]{5L,6L,9L});
		PrivacyLevel inputPrivacyLevel = PrivacyLevel.Private;
		constraints.put(inputDataRange, inputPrivacyLevel);
		Map<DataRange, PrivacyLevel> outputMap = constraints.getPrivacyLevel(new DataRange(new long[]{4L,4L,7L}, new long[]{5L,5L,8L}));
		assertTrue(outputMap.containsKey(inputDataRange));
		assertEquals(inputPrivacyLevel, outputMap.get(inputDataRange));
	}

	@Test
	public void getPrivacyLevelSingleConstraintInRangeTest(){
		DataRange inputDataRange = new DataRange(new long[]{3L,2L,7L}, new long[]{5L,6L,9L});
		PrivacyLevel inputPrivacyLevel = PrivacyLevel.Private;
		constraints.put(inputDataRange, inputPrivacyLevel);
		Map<DataRange, PrivacyLevel> outputMap = constraints.getPrivacyLevel(new DataRange(new long[]{1L,4L,7L}, new long[]{5L,5L,8L}));
		assertTrue(outputMap.containsKey(inputDataRange));
		assertEquals(inputPrivacyLevel, outputMap.get(inputDataRange));
	}

	@Test
	public void getPrivacyLevelSingleConstraintNotInRangeTest(){
		DataRange inputDataRange = new DataRange(new long[]{3L,2L,7L}, new long[]{5L,6L,9L});
		PrivacyLevel inputPrivacyLevel = PrivacyLevel.Private;
		constraints.put(inputDataRange, inputPrivacyLevel);
		Map<DataRange, PrivacyLevel> outputMap = constraints.getPrivacyLevel(new DataRange(new long[]{0L,4L,7L}, new long[]{2L,5L,8L}));
		assertFalse(outputMap.containsKey(inputDataRange));
	}

	@Test
	public void getPrivacyLevelMultiConstraintInRangeTest(){
		DataRange inputDataRange1 = new DataRange(new long[]{3L,2L,7L}, new long[]{5L,6L,9L});
		DataRange inputDataRange2 = new DataRange(new long[]{10L,14L,12L}, new long[]{45L,23L,15L});
		PrivacyLevel inputPrivacyLevel = PrivacyLevel.Private;
		constraints.put(inputDataRange1, inputPrivacyLevel);
		constraints.put(inputDataRange2, inputPrivacyLevel);
		Map<DataRange, PrivacyLevel> outputMap = constraints.getPrivacyLevel(new DataRange(new long[]{4L,3L,8L}, new long[]{50L,55L,19L}));
		assertTrue(outputMap.containsKey(inputDataRange1));
		assertTrue(outputMap.containsKey(inputDataRange2));
		assertEquals(inputPrivacyLevel, outputMap.get(inputDataRange1));
		assertEquals(inputPrivacyLevel, outputMap.get(inputDataRange2));
	}

	@Test
	public void getPrivacyLevelOfElementSingleConstraintTest(){
		DataRange inputDataRange = new DataRange(new long[]{3L,2L,7L}, new long[]{5L,6L,9L});
		PrivacyLevel inputPrivacyLevel = PrivacyLevel.Private;
		constraints.put(inputDataRange, inputPrivacyLevel);
		Map<DataRange, PrivacyLevel> outputMap = constraints.getPrivacyLevelOfElement(new long[]{4L,4L,7L});
		assertTrue(outputMap.containsKey(inputDataRange));
		assertEquals(inputPrivacyLevel, outputMap.get(inputDataRange));
	}

	@Test
	public void getPrivacyLevelOfElementOnLowerBoundSingleConstraintTest(){
		DataRange inputDataRange = new DataRange(new long[]{3L,2L,7L}, new long[]{5L,6L,9L});
		PrivacyLevel inputPrivacyLevel = PrivacyLevel.Private;
		constraints.put(inputDataRange, inputPrivacyLevel);
		Map<DataRange, PrivacyLevel> outputMap = constraints.getPrivacyLevelOfElement(new long[]{3L,2L,7L});
		assertTrue(outputMap.containsKey(inputDataRange));
		assertEquals(inputPrivacyLevel, outputMap.get(inputDataRange));
	}

	@Test
	public void getPrivacyLevelOfElementOnUpperBoundSingleConstraintTest(){
		DataRange inputDataRange = new DataRange(new long[]{3L,2L,7L}, new long[]{5L,6L,9L});
		PrivacyLevel inputPrivacyLevel = PrivacyLevel.Private;
		constraints.put(inputDataRange, inputPrivacyLevel);
		Map<DataRange, PrivacyLevel> outputMap = constraints.getPrivacyLevelOfElement(new long[]{5L,6L,9L});
		assertTrue(outputMap.containsKey(inputDataRange));
		assertEquals(inputPrivacyLevel, outputMap.get(inputDataRange));
	}

	@Test
	public void getPrivacyLevelOfElementSingleConstraintNotInRangeTest(){
		DataRange inputDataRange = new DataRange(new long[]{3L,2L,7L}, new long[]{5L,6L,9L});
		PrivacyLevel inputPrivacyLevel = PrivacyLevel.Private;
		constraints.put(inputDataRange, inputPrivacyLevel);
		Map<DataRange, PrivacyLevel> outputMap = constraints.getPrivacyLevelOfElement(new long[]{7L,10L,1L});
		assertFalse(outputMap.containsKey(inputDataRange));
	}

	@Test
	public void getPrivacyLevelOfElementDoubleConstraintInSingleTest(){
		DataRange inputDataRange1 = new DataRange(new long[]{3L,2L,7L}, new long[]{5L,6L,9L});
		DataRange inputDataRange2 = new DataRange(new long[]{10L,14L,12L}, new long[]{45L,23L,15L});
		PrivacyLevel inputPrivacyLevel = PrivacyLevel.Private;
		constraints.put(inputDataRange1, inputPrivacyLevel);
		constraints.put(inputDataRange2, inputPrivacyLevel);
		Map<DataRange, PrivacyLevel> outputMap = constraints.getPrivacyLevelOfElement(new long[]{21L,17L,13L});
		assertTrue("inputDataRange2 should be in outputMap since element is in the range", outputMap.containsKey(inputDataRange2));
		assertFalse("inputDataRange1 should not be in outputMap", outputMap.containsKey(inputDataRange1));
		assertEquals(inputPrivacyLevel, outputMap.get(inputDataRange2));
	}

	@Test
	public void getPrivacyLevelOfElementDoubleConstraintInBothTest(){
		DataRange inputDataRange1 = new DataRange(new long[]{3L,2L,7L}, new long[]{5L,6L,9L});
		DataRange inputDataRange2 = new DataRange(new long[]{1,1L,8L}, new long[]{45L,23L,15L});
		PrivacyLevel inputPrivacyLevel = PrivacyLevel.Private;
		constraints.put(inputDataRange1, inputPrivacyLevel);
		constraints.put(inputDataRange2, inputPrivacyLevel);
		Map<DataRange, PrivacyLevel> outputMap = constraints.getPrivacyLevelOfElement(new long[]{4L,4L,9L});
		assertTrue("inputDataRange2 should be in outputMap since element is in the range", outputMap.containsKey(inputDataRange2));
		assertTrue("inputDataRange1 should be in outputMap since element is in the range", outputMap.containsKey(inputDataRange1));
		assertEquals(inputPrivacyLevel, outputMap.get(inputDataRange2));
		assertEquals(inputPrivacyLevel, outputMap.get(inputDataRange1));
	}

	@Test
	public void getDataRangesOfPrivacyLevelTest(){
		DataRange inputDataRange1 = new DataRange(new long[]{60L,30L,70L}, new long[]{90L,60L,150L});
		DataRange inputDataRange2 = new DataRange(new long[]{10,10L,18L}, new long[]{450L,230L,250L});
		DataRange inputDataRange3 = new DataRange(new long[]{300L,250L,740L}, new long[]{520L,630L,1090L});
		DataRange inputDataRange4 = new DataRange(new long[]{10,10L,10L}, new long[]{30L,40L,50L});
		PrivacyLevel inputPrivacyLevel1 = PrivacyLevel.Private;
		PrivacyLevel inputPrivacyLevel2 = PrivacyLevel.PrivateAggregation;
		constraints.put(inputDataRange1, inputPrivacyLevel1);
		constraints.put(inputDataRange2, inputPrivacyLevel1);
		constraints.put(inputDataRange3, inputPrivacyLevel2);
		constraints.put(inputDataRange4, inputPrivacyLevel2);
		DataRange[] outputDataRanges1 = constraints.getDataRangesOfPrivacyLevel(inputPrivacyLevel1);
		assertEquals(inputDataRange1, outputDataRanges1[0]);
		assertEquals(inputDataRange2, outputDataRanges1[1]);
		DataRange[] outputDataRanges2 = constraints.getDataRangesOfPrivacyLevel(inputPrivacyLevel2);
		assertEquals(inputDataRange3, outputDataRanges2[0]);
		assertEquals(inputDataRange4, outputDataRanges2[1]);
	}
}
