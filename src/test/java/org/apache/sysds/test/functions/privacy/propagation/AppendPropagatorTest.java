///*
// * Licensed to the Apache Software Foundation (ASF) under one
// * or more contributor license agreements.  See the NOTICE file
// * distributed with this work for additional information
// * regarding copyright ownership.  The ASF licenses this file
// * to you under the Apache License, Version 2.0 (the
// * "License"); you may not use this file except in compliance
// * with the License.  You may obtain a copy of the License at
// *
// *   http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing,
// * software distributed under the License is distributed on an
// * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// * KIND, either express or implied.  See the License for the
// * specific language governing permissions and limitations
// * under the License.
// */
//
//package org.apache.sysds.test.functions.privacy.propagation;
//
//import org.apache.sysds.runtime.instructions.cp.Data;
//import org.apache.sysds.runtime.instructions.cp.DoubleObject;
//import org.apache.sysds.runtime.instructions.cp.IntObject;
//import org.apache.sysds.runtime.instructions.cp.ListObject;
//import org.apache.sysds.runtime.instructions.cp.ScalarObject;
//import org.apache.sysds.runtime.matrix.data.MatrixBlock;
//import org.apache.sysds.runtime.meta.MatrixCharacteristics;
//import org.apache.sysds.runtime.privacy.PrivacyConstraint;
//import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
//import org.apache.sysds.runtime.privacy.finegrained.DataRange;
//import org.apache.sysds.runtime.privacy.propagation.AppendPropagator;
//import org.apache.sysds.runtime.privacy.propagation.CBindPropagator;
//import org.apache.sysds.runtime.privacy.propagation.ListAppendPropagator;
//import org.apache.sysds.runtime.privacy.propagation.ListRemovePropagator;
//import org.apache.sysds.runtime.privacy.propagation.Propagator;
//import org.apache.sysds.runtime.privacy.propagation.PropagatorMultiReturn;
//import org.apache.sysds.runtime.privacy.propagation.RBindPropagator;
//import org.apache.sysds.test.AutomatedTestBase;
//import org.apache.sysds.test.TestConfiguration;
//import org.apache.sysds.test.TestUtils;
//import org.junit.Assert;
//import org.junit.Ignore;
//import org.junit.Test;
//
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.List;
//import java.util.Map;
//
//public class AppendPropagatorTest extends AutomatedTestBase {
//
//	private final static String TEST_DIR = "functions/privacy/";
//	private final static String TEST_NAME_RBIND = "RBindTest";
//	private final static String TEST_NAME_CBIND = "CBindTest";
//	private final static String TEST_NAME_STRING = "StringAppendTest";
//	private final static String TEST_NAME_LIST = "ListAppendTest";
//	private final static String TEST_CLASS_DIR = TEST_DIR + AppendPropagatorTest.class.getSimpleName() + "/";
//
//	@Override public void setUp() {
//		TestUtils.clearAssertionInformation();
//		addTestConfiguration(TEST_NAME_RBIND, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_RBIND, new String[] {"C"}));
//		addTestConfiguration(TEST_NAME_CBIND, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_CBIND, new String[] {"C"}));
//		addTestConfiguration(TEST_NAME_STRING, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_STRING, new String[] {"C"}));
//		addTestConfiguration(TEST_NAME_LIST, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_LIST, new String[] {"C"}));
//	}
//
//	@Test
//	public void generalOnlyRBindPrivate1Test(){
//		generalOnlyRBindTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint());
//	}
//
//	@Test
//	public void generalOnlyRBindPrivateAggregation1Test(){
//		generalOnlyRBindTest(new PrivacyConstraint(PrivacyLevel.PrivateAggregation), new PrivacyConstraint());
//	}
//
//	@Test
//	public void generalOnlyRBindNoneTest(){
//		generalOnlyRBindTest(new PrivacyConstraint(), new PrivacyConstraint());
//	}
//
//	@Test
//	public void generalOnlyRBindPrivate2Test(){
//		generalOnlyRBindTest(new PrivacyConstraint(), new PrivacyConstraint(PrivacyLevel.Private));
//	}
//
//	@Test
//	public void generalOnlyRBindPrivateAggregation2Test(){
//		generalOnlyRBindTest(new PrivacyConstraint(), new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
//	}
//
//	@Test
//	public void generalOnlyRBindPrivatePrivateTest(){
//		generalOnlyRBindTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint(PrivacyLevel.Private));
//	}
//
//	@Test
//	public void generalOnlyRBindPrivatePrivateAggregationTest(){
//		generalOnlyRBindTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
//	}
//
//	@Test
//	public void generalOnlyCBindPrivate1Test(){
//		generalOnlyCBindTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint());
//	}
//
//	@Test
//	public void generalOnlyCBindPrivateAggregation1Test(){
//		generalOnlyCBindTest(new PrivacyConstraint(PrivacyLevel.PrivateAggregation), new PrivacyConstraint());
//	}
//
//	@Test
//	public void generalOnlyCBindNoneTest(){
//		generalOnlyCBindTest(new PrivacyConstraint(), new PrivacyConstraint());
//	}
//
//	@Test
//	public void generalOnlyCBindPrivate2Test(){
//		generalOnlyCBindTest(new PrivacyConstraint(), new PrivacyConstraint(PrivacyLevel.Private));
//	}
//
//	@Test
//	public void generalOnlyCBindPrivateAggregation2Test(){
//		generalOnlyCBindTest(new PrivacyConstraint(), new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
//	}
//
//	@Test
//	public void generalOnlyCBindPrivatePrivateTest(){
//		generalOnlyCBindTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint(PrivacyLevel.Private));
//	}
//
//	@Test
//	public void generalOnlyCBindPrivatePrivateAggregationTest(){
//		generalOnlyCBindTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
//	}
//
//	@Test
//	public void generalOnlyListAppendPrivate1Test(){
//		generalOnlyListAppendTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint());
//	}
//
//	@Test
//	public void generalOnlyListAppendPrivateAggregation1Test(){
//		generalOnlyCBindTest(new PrivacyConstraint(PrivacyLevel.PrivateAggregation), new PrivacyConstraint());
//	}
//
//	@Test
//	public void generalOnlyListAppendNoneTest(){
//		generalOnlyListAppendTest(new PrivacyConstraint(), new PrivacyConstraint());
//	}
//
//	@Test
//	public void generalOnlyListAppendPrivate2Test(){
//		generalOnlyListAppendTest(new PrivacyConstraint(), new PrivacyConstraint(PrivacyLevel.Private));
//	}
//
//	@Test
//	public void generalOnlyListAppendPrivateAggregation2Test(){
//		generalOnlyListAppendTest(new PrivacyConstraint(), new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
//	}
//
//	@Test
//	public void generalOnlyListAppendPrivatePrivateTest(){
//		generalOnlyListAppendTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint(PrivacyLevel.Private));
//	}
//
//	@Test
//	public void generalOnlyListAppendPrivatePrivateAggregationTest(){
//		generalOnlyListAppendTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
//	}
//
//	@Test
//	public void generalOnlyListRemoveAppendPrivate1Test(){
//		generalOnlyListRemoveAppendTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint(),
//			PrivacyLevel.Private, PrivacyLevel.Private);
//	}
//
//	@Test
//	public void generalOnlyListRemoveAppendPrivateAggregation1Test(){
//		generalOnlyListRemoveAppendTest(new PrivacyConstraint(PrivacyLevel.PrivateAggregation), new PrivacyConstraint(),
//			PrivacyLevel.PrivateAggregation, PrivacyLevel.PrivateAggregation);
//	}
//
//	@Test
//	public void generalOnlyListRemoveAppendNoneTest(){
//		generalOnlyListRemoveAppendTest(new PrivacyConstraint(), new PrivacyConstraint(),
//			PrivacyLevel.None, PrivacyLevel.None);
//	}
//
//	@Test
//	public void generalOnlyListRemoveAppendPrivate2Test(){
//		generalOnlyListRemoveAppendTest(new PrivacyConstraint(), new PrivacyConstraint(PrivacyLevel.Private),
//			PrivacyLevel.Private, PrivacyLevel.Private);
//	}
//
//	@Test
//	public void generalOnlyListRemoveAppendPrivateAggregation2Test(){
//		generalOnlyListRemoveAppendTest(new PrivacyConstraint(), new PrivacyConstraint(PrivacyLevel.PrivateAggregation),
//			PrivacyLevel.PrivateAggregation, PrivacyLevel.PrivateAggregation);
//	}
//
//	@Test
//	public void generalOnlyListRemoveAppendPrivatePrivateTest(){
//		generalOnlyListRemoveAppendTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint(PrivacyLevel.Private),
//			PrivacyLevel.Private, PrivacyLevel.Private);
//	}
//
//	@Test
//	public void generalOnlyListRemoveAppendPrivatePrivateAggregationTest(){
//		generalOnlyListRemoveAppendTest(new PrivacyConstraint(PrivacyLevel.Private), new PrivacyConstraint(PrivacyLevel.PrivateAggregation),
//			PrivacyLevel.Private, PrivacyLevel.Private);
//	}
//
//	@Test
//	public void finegrainedRBindPrivate1(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		finegrainedRBindTest(constraint1, constraint2);
//	}
//
//	@Test
//	public void finegrainedRBindPrivateAndPrivateAggregate1(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0},new long[]{3,1}), PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		finegrainedRBindTest(constraint1, constraint2);
//	}
//
//	@Test
//	public void finegrainedRBindPrivate2(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		finegrainedRBindTest(constraint1, constraint2);
//	}
//
//	@Test
//	public void finegrainedRBindPrivateAndPrivateAggregate2(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0},new long[]{3,1}), PrivacyLevel.PrivateAggregation);
//		finegrainedRBindTest(constraint1, constraint2);
//	}
//
//	@Test
//	public void finegrainedRBindPrivate12(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		finegrainedRBindTest(constraint1, constraint2);
//	}
//
//	@Test
//	public void finegrainedRBindPrivateAndPrivateAggregate12(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0},new long[]{0,1}), PrivacyLevel.Private);
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0},new long[]{3,1}), PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0},new long[]{0,1}), PrivacyLevel.Private);
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{2,0}), PrivacyLevel.PrivateAggregation);
//		finegrainedRBindTest(constraint1, constraint2);
//	}
//
//	@Test
//	public void finegrainedCBindPrivate1(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		finegrainedCBindTest(constraint1, constraint2);
//	}
//
//	@Test
//	public void finegrainedCBindPrivateAndPrivateAggregate1(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0},new long[]{3,1}), PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		finegrainedCBindTest(constraint1, constraint2);
//	}
//
//	@Test
//	public void finegrainedCBindPrivate2(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		finegrainedCBindTest(constraint1, constraint2);
//	}
//
//	@Test
//	public void finegrainedCBindPrivateAndPrivateAggregate2(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0},new long[]{3,1}), PrivacyLevel.PrivateAggregation);
//		finegrainedCBindTest(constraint1, constraint2);
//	}
//
//	@Test
//	public void finegrainedCBindPrivate12(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		finegrainedCBindTest(constraint1, constraint2);
//	}
//
//	@Test
//	public void finegrainedCBindPrivateAndPrivateAggregate12(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0},new long[]{0,1}), PrivacyLevel.Private);
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0},new long[]{3,1}), PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0},new long[]{0,1}), PrivacyLevel.Private);
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{2,0}), PrivacyLevel.PrivateAggregation);
//		finegrainedCBindTest(constraint1, constraint2);
//	}
//
//	@Test
//	public void finegrainedListAppendPrivate1(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1},new long[]{2}), PrivacyLevel.Private);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		finegrainedListAppendTest(constraint1, constraint2);
//	}
//
//	@Test
//	public void finegrainedListAppendPrivateAndPrivateAggregate1(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1},new long[]{2}), PrivacyLevel.Private);
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{3},new long[]{5}), PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		finegrainedListAppendTest(constraint1, constraint2);
//	}
//
//	@Test
//	public void finegrainedListAppendPrivate2(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1},new long[]{2}), PrivacyLevel.Private);
//		finegrainedListAppendTest(constraint1, constraint2);
//	}
//
//	@Test
//	public void finegrainedListAppendPrivateAndPrivateAggregate2(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1},new long[]{2}), PrivacyLevel.Private);
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{3},new long[]{5}), PrivacyLevel.PrivateAggregation);
//		finegrainedListAppendTest(constraint1, constraint2);
//	}
//
//	@Test
//	public void finegrainedListAppendPrivate12(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1},new long[]{2}), PrivacyLevel.Private);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1},new long[]{4}), PrivacyLevel.Private);
//		finegrainedListAppendTest(constraint1, constraint2);
//	}
//
//	@Test
//	public void finegrainedListAppendPrivateAndPrivateAggregate12(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{0},new long[]{0}), PrivacyLevel.Private);
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{2},new long[]{3}), PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{0},new long[]{1}), PrivacyLevel.Private);
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{3},new long[]{5}), PrivacyLevel.PrivateAggregation);
//		finegrainedListAppendTest(constraint1, constraint2);
//	}
//
//	@Test
//	public void testFunction(){
//		int dataLength = 9;
//		List<Data> dataList = new ArrayList<>();
//		for ( int i = 0; i < dataLength; i++)
//			dataList.add(new DoubleObject(i));
//		ListObject l = new ListObject(dataList);
//		ListObject lCopy = l.copy();
//		int position = 4;
//		ListObject output = l.remove(position);
//		Assert.assertEquals(lCopy.getData(position), output.getData().get(0));
//	}
//
//	@Test
//	public void finegrainedListRemoveAppendNone1(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1},new long[]{2}), PrivacyLevel.Private);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		finegrainedListRemoveAppendTest(constraint1, constraint2, PrivacyLevel.None);
//	}
//
//	@Test
//	public void finegrainedListRemoveAppendPrivate1(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1},new long[]{6}), PrivacyLevel.Private);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		finegrainedListRemoveAppendTest(constraint1, constraint2, PrivacyLevel.Private);
//	}
//
//	@Test
//	public void finegrainedListRemoveAppendPrivate2(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1},new long[]{6}), PrivacyLevel.Private);
//		PrivacyConstraint constraint2 = new PrivacyConstraint(PrivacyLevel.PrivateAggregation);
//		finegrainedListRemoveAppendTest(constraint1, constraint2, PrivacyLevel.Private);
//	}
//
//	@Test
//	public void finegrainedListRemoveAppendPrivate3(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.Private);
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1},new long[]{6}), PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		finegrainedListRemoveAppendTest(constraint1, constraint2, PrivacyLevel.Private);
//	}
//
//	@Test
//	public void finegrainedListRemoveAppendPrivate4(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.PrivateAggregation);
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{4},new long[]{4}), PrivacyLevel.Private);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		finegrainedListRemoveAppendTest(constraint1, constraint2, PrivacyLevel.Private, true);
//	}
//
//	@Test
//	public void finegrainedListRemoveAppendPrivateAndPrivateAggregate1(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1},new long[]{2}), PrivacyLevel.Private);
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{3},new long[]{5}), PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		finegrainedListRemoveAppendTest(constraint1, constraint2, PrivacyLevel.PrivateAggregation);
//	}
//
//	@Test
//	public void finegrainedListRemoveAppendPrivateAggregate2(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint(PrivacyLevel.PrivateAggregation);
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{5},new long[]{8}), PrivacyLevel.Private);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		finegrainedListRemoveAppendTest(constraint1, constraint2, PrivacyLevel.PrivateAggregation);
//	}
//
//	@Test
//	public void integrationRBindTestNoneNone(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pcExpected = new PrivacyConstraint(PrivacyLevel.None);
//		integrationRBindTest(pc1, pc2, pcExpected);
//	}
//
//	@Test
//	public void integrationRBindTestPrivateNone(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.Private);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pcExpected = new PrivacyConstraint();
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0}, new long[]{19,9}), PrivacyLevel.Private);
//		integrationRBindTest(pc1, pc2, pcExpected);
//	}
//
//	@Test
//	public void integrationRBindTestPrivateAggregateNone(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pcExpected = new PrivacyConstraint();
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0}, new long[]{19,9}), PrivacyLevel.PrivateAggregation);
//		integrationRBindTest(pc1, pc2, pcExpected);
//	}
//
//	@Test
//	public void integrationRBindTestNonePrivateAggregate(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint pcExpected = new PrivacyConstraint();
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{20,0}, new long[]{49, 9}), PrivacyLevel.PrivateAggregation);
//		integrationRBindTest(pc1, pc2, pcExpected);
//	}
//
//	@Test
//	public void integrationRBindTestNonePrivate(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.Private);
//		PrivacyConstraint pcExpected = new PrivacyConstraint();
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{20,0}, new long[]{49, 9}), PrivacyLevel.Private);
//		integrationRBindTest(pc1, pc2, pcExpected);
//	}
//
//	@Test
//	public void integrationFinegrainedRBindPrivate1(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		integrationRBindTest(constraint1, constraint2, constraint1);
//	}
//
//	@Test
//	public void integrationFinegrainedRBindPrivateAndPrivateAggregate1(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0},new long[]{3,1}), PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		integrationRBindTest(constraint1, constraint2, constraint1);
//	}
//
//	@Test
//	public void integrationFinegrainedRBindPrivate2(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		PrivacyConstraint pcExcepted = new PrivacyConstraint();
//		pcExcepted.getFineGrainedPrivacy().put(new DataRange(new long[]{21,0}, new long[]{21,1}), PrivacyLevel.Private);
//		integrationRBindTest(constraint1, constraint2, pcExcepted);
//	}
//
//	@Test
//	public void integrationFinegrainedRBindPrivateAndPrivateAggregate2(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0},new long[]{3,1}), PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint pcExcepted = new PrivacyConstraint();
//		pcExcepted.getFineGrainedPrivacy().put(new DataRange(new long[]{21,0},new long[]{21,1}), PrivacyLevel.Private);
//		pcExcepted.getFineGrainedPrivacy().put(new DataRange(new long[]{22,0}, new long[]{23,1}), PrivacyLevel.PrivateAggregation);
//		integrationRBindTest(constraint1, constraint2, pcExcepted);
//	}
//
//	@Test
//	public void integrationFinegrainedRBindPrivate12(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		PrivacyConstraint pcExpected = new PrivacyConstraint();
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{21,0},new long[]{21,1}), PrivacyLevel.Private);
//		integrationRBindTest(constraint1, constraint2, pcExpected);
//	}
//
//	@Test
//	public void integrationFinegrainedRBindPrivateAndPrivateAggregate12(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0},new long[]{0,1}), PrivacyLevel.Private);
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0},new long[]{3,1}), PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0},new long[]{0,1}), PrivacyLevel.Private);
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{2,0}), PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint pcExpected = new PrivacyConstraint();
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0},new long[]{0,1}), PrivacyLevel.Private);
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{20,0},new long[]{20,1}), PrivacyLevel.Private);
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0}, new long[]{3,1}), PrivacyLevel.PrivateAggregation);
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{21,0}, new long[]{22,0}), PrivacyLevel.PrivateAggregation);
//		integrationRBindTest(constraint1, constraint2, pcExpected);
//	}
//
//	private void integrationRBindTest(PrivacyConstraint privacyConstraint1, PrivacyConstraint privacyConstraint2,
//		PrivacyConstraint expectedOutput){
//		TestConfiguration config = getAndLoadTestConfiguration(TEST_NAME_RBIND);
//		fullDMLScriptName = SCRIPT_DIR + TEST_DIR + config.getTestScript() + ".dml";
//
//		int rows1 = 20;
//		int rows2 = 30;
//		int cols = 10;
//		double[][] A = getRandomMatrix(rows1, cols, -10, 10, 0.5, 1);
//		double[][] B = getRandomMatrix(rows2, cols, -10, 10, 0.5, 1);
//		writeInputMatrixWithMTD("A", A, false, new MatrixCharacteristics(rows1, cols),  privacyConstraint1);
//		writeInputMatrixWithMTD("B", B, false, new MatrixCharacteristics(rows2, cols),  privacyConstraint2);
//
//		programArgs = new String[]{"-nvargs", "A=" + input("A"), "B=" + input("B"), "C=" + output("C")};
//		runTest(true,false,null,-1);
//
//		PrivacyConstraint outputConstraint = getPrivacyConstraintFromMetaData("C");
//		Assert.assertEquals(expectedOutput, outputConstraint);
//	}
//
//	@Test
//	public void integrationCBindTestNoneNone(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pcExpected = new PrivacyConstraint(PrivacyLevel.None);
//		integrationCBindTest(pc1, pc2, pcExpected);
//	}
//
//	@Test
//	public void integrationCBindTestPrivateNone(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.Private);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pcExpected = new PrivacyConstraint();
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0}, new long[]{9,19}), PrivacyLevel.Private);
//		integrationCBindTest(pc1, pc2, pcExpected);
//	}
//
//	@Test
//	public void integrationCBindTestPrivateAggregateNone(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pcExpected = new PrivacyConstraint();
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0}, new long[]{9,19}), PrivacyLevel.PrivateAggregation);
//		integrationCBindTest(pc1, pc2, pcExpected);
//	}
//
//	@Test
//	public void integrationCBindTestNonePrivateAggregate(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint pcExpected = new PrivacyConstraint();
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{0,20}, new long[]{9, 49}), PrivacyLevel.PrivateAggregation);
//		integrationCBindTest(pc1, pc2, pcExpected);
//	}
//
//	@Test
//	public void integrationCBindTestNonePrivate(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.Private);
//		PrivacyConstraint pcExpected = new PrivacyConstraint();
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{0,20}, new long[]{9, 49}), PrivacyLevel.Private);
//		integrationCBindTest(pc1, pc2, pcExpected);
//	}
//
//	@Test
//	public void integrationFinegrainedCBindPrivate1(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		integrationCBindTest(constraint1, constraint2, constraint1);
//	}
//
//	@Test
//	public void integrationFinegrainedCBindPrivateAndPrivateAggregate1(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0},new long[]{3,1}), PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		integrationCBindTest(constraint1, constraint2, constraint1);
//	}
//
//	@Test
//	public void integrationFinegrainedCBindPrivate2(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		PrivacyConstraint pcExcepted = new PrivacyConstraint();
//		pcExcepted.getFineGrainedPrivacy().put(new DataRange(new long[]{1,20}, new long[]{1,21}), PrivacyLevel.Private);
//		integrationCBindTest(constraint1, constraint2, pcExcepted);
//	}
//
//	@Test
//	public void integrationFinegrainedCBindPrivateAndPrivateAggregate2(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0},new long[]{3,1}), PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint pcExcepted = new PrivacyConstraint();
//		pcExcepted.getFineGrainedPrivacy().put(new DataRange(new long[]{1,20},new long[]{1,21}), PrivacyLevel.Private);
//		pcExcepted.getFineGrainedPrivacy().put(new DataRange(new long[]{2,20}, new long[]{3,21}), PrivacyLevel.PrivateAggregation);
//		integrationCBindTest(constraint1, constraint2, pcExcepted);
//	}
//
//	@Test
//	public void integrationFinegrainedCBindPrivate12(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		PrivacyConstraint pcExpected = new PrivacyConstraint();
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{1,1}), PrivacyLevel.Private);
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{1,20},new long[]{1,21}), PrivacyLevel.Private);
//		integrationCBindTest(constraint1, constraint2, pcExpected);
//	}
//
//	@Test
//	public void integrationFinegrainedCBindPrivateAndPrivateAggregate12(){
//		PrivacyConstraint constraint1 = new PrivacyConstraint();
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0},new long[]{0,1}), PrivacyLevel.Private);
//		constraint1.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0},new long[]{3,1}), PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint constraint2 = new PrivacyConstraint();
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0},new long[]{0,1}), PrivacyLevel.Private);
//		constraint2.getFineGrainedPrivacy().put(new DataRange(new long[]{1,0},new long[]{2,0}), PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint pcExpected = new PrivacyConstraint();
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{0,0},new long[]{0,1}), PrivacyLevel.Private);
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{0,20},new long[]{0,21}), PrivacyLevel.Private);
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{2,0}, new long[]{3,1}), PrivacyLevel.PrivateAggregation);
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{1,20}, new long[]{2,20}), PrivacyLevel.PrivateAggregation);
//		integrationCBindTest(constraint1, constraint2, pcExpected);
//	}
//
//	@Test
//	public void integrationStringAppendTestNoneNone(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.None);
//		integrationStringAppendTest(pc1, pc2, pc1);
//	}
//
//	@Test
//	public void integrationStringAppendTestPrivateNone(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.Private);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.None);
//		integrationStringAppendTest(pc1, pc2, pc1);
//	}
//
//	@Test
//	public void integrationStringAppendTestPrivateAggregateNone(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.None);
//		integrationStringAppendTest(pc1, pc2, pc1);
//	}
//
//	@Test
//	public void integrationStringAppendTestNonePrivateAggregate(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.PrivateAggregation);
//		integrationStringAppendTest(pc1, pc2, pc2);
//	}
//
//	@Test
//	public void integrationStringAppendTestNonePrivate(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.Private);
//		integrationStringAppendTest(pc1, pc2, pc2);
//	}
//
//	@Test
//	public void integrationListAppendTestNoneNone(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.None);
//		integrationListAppendTest(pc1, pc2, pc1);
//	}
//
//	@Test
//	public void integrationListAppendTestPrivateNone(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.Private);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pcExpected = new PrivacyConstraint();
//		pcExpected.getFineGrainedPrivacy().put(new DataRange(new long[]{0}, new long[]{0}), PrivacyLevel.Private);
//		integrationListAppendTest(pc1, pc2, pcExpected);
//	}
//
//	@Test
//	public void integrationListAppendTestPrivateAggregateNone(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.PrivateAggregation);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.None);
//		integrationListAppendTest(pc1, pc2, pc1);
//	}
//
//	@Test
//	public void integrationListAppendTestNonePrivateAggregate(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.PrivateAggregation);
//		integrationListAppendTest(pc1, pc2, pc2);
//	}
//
//	@Test
//	public void integrationListAppendTestNonePrivate(){
//		PrivacyConstraint pc1 = new PrivacyConstraint(PrivacyLevel.None);
//		PrivacyConstraint pc2 = new PrivacyConstraint(PrivacyLevel.Private);
//		integrationListAppendTest(pc1, pc2, pc2);
//	}
//
//	private static void generalOnlyRBindTest(PrivacyConstraint constraint1, PrivacyConstraint constraint2){
//		int columns = 2;
//		int rows1 = 4;
//		int rows2 = 3;
//		MatrixBlock inputMatrix1 = new MatrixBlock(rows1,columns,3);
//		MatrixBlock inputMatrix2 = new MatrixBlock(rows2,columns,4);
//		AppendPropagator propagator = new RBindPropagator(inputMatrix1, constraint1, inputMatrix2, constraint2);
//		PrivacyConstraint mergedConstraint = propagator.propagate();
//		Assert.assertEquals(mergedConstraint.getPrivacyLevel(), PrivacyLevel.None);
//		Map<DataRange, PrivacyLevel> firstHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
//			new DataRange(new long[]{0,0}, new long[]{rows1-1,columns-1}));
//		firstHalfPrivacy.forEach((range,level) -> Assert.assertEquals(constraint1.getPrivacyLevel(),level));
//		Map<DataRange, PrivacyLevel> secondHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
//			new DataRange(new long[]{rows1,0}, new long[]{rows1+rows2-1,columns-1}));
//		secondHalfPrivacy.forEach((range,level) -> Assert.assertEquals(constraint2.getPrivacyLevel(),level));
//	}
//
//	private static void generalOnlyCBindTest(PrivacyConstraint constraint1, PrivacyConstraint constraint2){
//		int rows = 2;
//		int columns1 = 4;
//		int columns2 = 3;
//		MatrixBlock inputMatrix1 = new MatrixBlock(rows,columns1,3);
//		MatrixBlock inputMatrix2 = new MatrixBlock(rows,columns2,4);
//		AppendPropagator propagator = new CBindPropagator(inputMatrix1, constraint1, inputMatrix2, constraint2);
//		PrivacyConstraint mergedConstraint = propagator.propagate();
//		Assert.assertEquals(mergedConstraint.getPrivacyLevel(), PrivacyLevel.None);
//		Map<DataRange, PrivacyLevel> firstHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
//			new DataRange(new long[]{0,0}, new long[]{rows-1,columns1-1}));
//		firstHalfPrivacy.forEach((range,level) -> Assert.assertEquals(constraint1.getPrivacyLevel(),level));
//		Map<DataRange, PrivacyLevel> secondHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
//			new DataRange(new long[]{0,columns1}, new long[]{rows,columns1+columns2-1}));
//		secondHalfPrivacy.forEach((range,level) -> Assert.assertEquals(constraint2.getPrivacyLevel(),level));
//	}
//
//	private static void generalOnlyListAppendTest(PrivacyConstraint constraint1, PrivacyConstraint constraint2){
//		int length1 = 6;
//		List<Data> dataList1 = Arrays.asList(new Data[length1]);
//		ListObject input1 = new ListObject(dataList1);
//		int length2 = 11;
//		List<Data> dataList2 = Arrays.asList(new Data[length2]);
//		ListObject input2 = new ListObject(dataList2);
//		Propagator propagator = new ListAppendPropagator(input1, constraint1, input2, constraint2);
//		PrivacyConstraint mergedConstraint = propagator.propagate();
//		Map<DataRange, PrivacyLevel> firstHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
//			new DataRange(new long[]{0}, new long[]{length1-1})
//		);
//		firstHalfPrivacy.forEach((range,level) -> Assert.assertEquals(constraint1.getPrivacyLevel(),level));
//		Map<DataRange, PrivacyLevel> secondHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
//			new DataRange(new long[length1], new long[]{length1+length2-1})
//		);
//		secondHalfPrivacy.forEach((range,level) -> Assert.assertEquals(constraint2.getPrivacyLevel(),level));
//	}
//
//	private static void generalOnlyListRemoveAppendTest(
//		PrivacyConstraint constraint1, PrivacyConstraint constraint2, PrivacyLevel expected1, PrivacyLevel expected2){
//		int dataLength = 9;
//		List<Data> dataList = new ArrayList<>();
//		for ( int i = 0; i < dataLength; i++){
//			dataList.add(new DoubleObject(i));
//		}
//		ListObject inputList = new ListObject(dataList);
//
//		int removePositionInt = 5;
//		ScalarObject removePosition = new IntObject(removePositionInt);
//
//		PropagatorMultiReturn propagator = new ListRemovePropagator(inputList, constraint1, removePosition, constraint2);
//		PrivacyConstraint[] mergedConstraints = propagator.propagate();
//
//		Assert.assertEquals(expected1, mergedConstraints[0].getPrivacyLevel());
//		Assert.assertEquals(expected2, mergedConstraints[1].getPrivacyLevel());
//		Assert.assertFalse("The first output constraint should have no fine-grained constraints", mergedConstraints[0].hasFineGrainedConstraints());
//		Assert.assertFalse("The second output constraint should have no fine-grained constraints", mergedConstraints[1].hasFineGrainedConstraints());
//	}
//
//	private static void finegrainedRBindTest(PrivacyConstraint constraint1, PrivacyConstraint constraint2){
//		int columns = 2;
//		int rows1 = 4;
//		int rows2 = 3;
//		MatrixBlock inputMatrix1 = new MatrixBlock(rows1,columns,3);
//		MatrixBlock inputMatrix2 = new MatrixBlock(rows2,columns,4);
//		AppendPropagator propagator = new RBindPropagator(inputMatrix1, constraint1, inputMatrix2, constraint2);
//		PrivacyConstraint mergedConstraint = propagator.propagate();
//		Assert.assertEquals(mergedConstraint.getPrivacyLevel(), PrivacyLevel.None);
//		Map<DataRange, PrivacyLevel> firstHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
//			new DataRange(new long[]{0,0}, new long[]{rows1-1,columns-1}));
//		constraint1.getFineGrainedPrivacy().getAllConstraintsList().forEach(
//			constraint -> Assert.assertTrue("Merged constraint should contain same privacy levels as input 1",
//				firstHalfPrivacy.containsValue(constraint.getValue()))
//		);
//		Map<DataRange, PrivacyLevel> secondHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
//			new DataRange(new long[]{rows1,0}, new long[]{rows1+rows2-1,columns-1}));
//		constraint2.getFineGrainedPrivacy().getAllConstraintsList().forEach(
//			constraint -> Assert.assertTrue("Merged constraint should contain same privacy levels as input 2",
//				secondHalfPrivacy.containsValue(constraint.getValue()))
//		);
//	}
//
//	private static void finegrainedCBindTest(PrivacyConstraint constraint1, PrivacyConstraint constraint2){
//		int rows = 6;
//		int columns1 = 4;
//		int columns2 = 3;
//		MatrixBlock inputMatrix1 = new MatrixBlock(rows,columns1,3);
//		MatrixBlock inputMatrix2 = new MatrixBlock(rows,columns2,4);
//		AppendPropagator propagator = new CBindPropagator(inputMatrix1, constraint1, inputMatrix2, constraint2);
//		PrivacyConstraint mergedConstraint = propagator.propagate();
//		Assert.assertEquals(mergedConstraint.getPrivacyLevel(), PrivacyLevel.None);
//		Map<DataRange, PrivacyLevel> firstHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
//			new DataRange(new long[]{0,0}, new long[]{rows-1,columns1-1}));
//		constraint1.getFineGrainedPrivacy().getAllConstraintsList().forEach(
//			constraint -> Assert.assertTrue("Merged constraint should contain same privacy levels as input 1",
//				firstHalfPrivacy.containsValue(constraint.getValue()))
//		);
//		Map<DataRange, PrivacyLevel> secondHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
//			new DataRange(new long[]{0,columns1}, new long[]{rows,columns1+columns2-1}));
//		constraint2.getFineGrainedPrivacy().getAllConstraintsList().forEach(
//			constraint -> Assert.assertTrue("Merged constraint should contain same privacy levels as input 2",
//				secondHalfPrivacy.containsValue(constraint.getValue()))
//		);
//	}
//
//	private static void finegrainedListAppendTest(PrivacyConstraint constraint1, PrivacyConstraint constraint2){
//		int length1 = 6;
//		List<Data> dataList1 = Arrays.asList(new Data[length1]);
//		ListObject input1 = new ListObject(dataList1);
//		int length2 = 11;
//		List<Data> dataList2 = Arrays.asList(new Data[length2]);
//		ListObject input2 = new ListObject(dataList2);
//		Propagator propagator = new ListAppendPropagator(input1, constraint1, input2, constraint2);
//		PrivacyConstraint mergedConstraint = propagator.propagate();
//		Assert.assertEquals(mergedConstraint.getPrivacyLevel(), PrivacyLevel.None);
//		Map<DataRange, PrivacyLevel> firstHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
//			new DataRange(new long[]{0}, new long[]{length1-1})
//		);
//		constraint1.getFineGrainedPrivacy().getAllConstraintsList().forEach(
//			constraint -> Assert.assertTrue("Merged constraint should contain same privacy levels as input 1",
//				firstHalfPrivacy.containsValue(constraint.getValue()))
//		);
//		Map<DataRange, PrivacyLevel> secondHalfPrivacy = mergedConstraint.getFineGrainedPrivacy().getPrivacyLevel(
//			new DataRange(new long[]{length1}, new long[]{length1+length2-1})
//		);
//		constraint2.getFineGrainedPrivacy().getAllConstraintsList().forEach(
//			constraint -> Assert.assertTrue("Merged constraint should contain same privacy levels as input 2",
//				secondHalfPrivacy.containsValue(constraint.getValue()))
//		);
//	}
//
//	private static void finegrainedListRemoveAppendTest(
//		PrivacyConstraint constraint1, PrivacyConstraint constraint2, PrivacyLevel expectedOutput2){
//		finegrainedListRemoveAppendTest(constraint1, constraint2, expectedOutput2, false);
//	}
//
//	private static void finegrainedListRemoveAppendTest(
//		PrivacyConstraint constraint1, PrivacyConstraint constraint2, PrivacyLevel expectedOutput2, boolean singleElementPrivacy){
//		int dataLength = 9;
//		List<Data> dataList = new ArrayList<>();
//		for ( int i = 0; i < dataLength; i++){
//			dataList.add(new DoubleObject(i));
//		}
//		ListObject inputList = new ListObject(dataList);
//		int removePositionInt = 5;
//		ScalarObject removePosition = new IntObject(removePositionInt);
//		PropagatorMultiReturn propagator = new ListRemovePropagator(inputList, constraint1, removePosition, constraint2);
//		PrivacyConstraint[] mergedConstraints = propagator.propagate();
//
//		if ( !singleElementPrivacy ){
//			Map<DataRange, PrivacyLevel> outputPrivacy = mergedConstraints[0].getFineGrainedPrivacy().getPrivacyLevel(
//				new DataRange(new long[]{0}, new long[]{dataLength-1})
//			);
//			constraint1.getFineGrainedPrivacy().getAllConstraintsList().forEach(
//				constraint -> Assert.assertTrue("Merged constraint should contain same privacy levels as input 1",
//					outputPrivacy.containsValue(constraint.getValue()))
//			);
//		}
//
//		Assert.assertEquals(expectedOutput2, mergedConstraints[1].getPrivacyLevel());
//		Assert.assertFalse(mergedConstraints[1].hasFineGrainedConstraints());
//	}
//
//	private void integrationCBindTest(PrivacyConstraint privacyConstraint1, PrivacyConstraint privacyConstraint2,
//		PrivacyConstraint expectedOutput){
//		TestConfiguration config = getAndLoadTestConfiguration(TEST_NAME_CBIND);
//		fullDMLScriptName = SCRIPT_DIR + TEST_DIR + config.getTestScript() + ".dml";
//
//		int cols1 = 20;
//		int cols2 = 30;
//		int rows = 10;
//		double[][] A = getRandomMatrix(rows, cols1, -10, 10, 0.5, 1);
//		double[][] B = getRandomMatrix(rows, cols2, -10, 10, 0.5, 1);
//		writeInputMatrixWithMTD("A", A, false, new MatrixCharacteristics(rows, cols1),  privacyConstraint1);
//		writeInputMatrixWithMTD("B", B, false, new MatrixCharacteristics(rows, cols2),  privacyConstraint2);
//
//		programArgs = new String[]{"-nvargs", "A=" + input("A"), "B=" + input("B"), "C=" + output("C")};
//		runTest(true,false,null,-1);
//
//		PrivacyConstraint outputConstraint = getPrivacyConstraintFromMetaData("C");
//		Assert.assertEquals(expectedOutput, outputConstraint);
//	}
//
//	private void integrationStringAppendTest(PrivacyConstraint privacyConstraint1, PrivacyConstraint privacyConstraint2,
//		PrivacyConstraint expectedOutput){
//		TestConfiguration config = getAndLoadTestConfiguration(TEST_NAME_STRING);
//		fullDMLScriptName = SCRIPT_DIR + TEST_DIR + config.getTestScript() + ".dml";
//
//		int cols = 1;
//		int rows = 1;
//		double[][] A = getRandomMatrix(rows, cols, -10, 10, 0.5, 1);
//		double[][] B = getRandomMatrix(rows, cols, -10, 10, 0.5, 1);
//		writeInputMatrixWithMTD("A", A, false, new MatrixCharacteristics(rows, cols),  privacyConstraint1);
//		writeInputMatrixWithMTD("B", B, false, new MatrixCharacteristics(rows, cols),  privacyConstraint2);
//
//		programArgs = new String[]{"-nvargs", "A=" + input("A"), "B=" + input("B"), "C=" + output("C")};
//		runTest(true,false,null,-1);
//
//		PrivacyConstraint outputConstraint = getPrivacyConstraintFromMetaData("C");
//		Assert.assertEquals(expectedOutput, outputConstraint);
//	}
//
//	private void integrationListAppendTest(PrivacyConstraint privacyConstraint1, PrivacyConstraint privacyConstraint2,
//		PrivacyConstraint expectedOutput){
//		TestConfiguration config = getAndLoadTestConfiguration(TEST_NAME_LIST);
//		fullDMLScriptName = SCRIPT_DIR + TEST_DIR + config.getTestScript() + ".dml";
//
//		int cols = 1;
//		int rows = 5;
//		double[][] A = getRandomMatrix(rows, cols, -10, 10, 0.5, 1);
//		double[][] B = getRandomMatrix(rows, cols, -10, 10, 0.5, 1);
//		writeInputMatrixWithMTD("A", A, false, new MatrixCharacteristics(rows, cols),  privacyConstraint1);
//		writeInputMatrixWithMTD("B", B, false, new MatrixCharacteristics(rows, cols),  privacyConstraint2);
//
//		programArgs = new String[]{"-nvargs", "A=" + input("A"), "B=" + input("B"), "C=" + output("C")};
//		runTest(true,false,null,-1);
//
//		PrivacyConstraint outputConstraint = getPrivacyConstraintFromMetaData("C");
//		Assert.assertEquals(expectedOutput, outputConstraint);
//	}
//}
