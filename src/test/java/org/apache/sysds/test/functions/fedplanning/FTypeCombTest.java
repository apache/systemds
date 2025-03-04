package org.apache.sysds.test.functions.fedplanning;
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
//package org.apache.sysds.test.functions.privacy.fedplanning;
//
//import org.apache.sysds.hops.fedplanner.FTypes.FType;
//import org.apache.sysds.hops.fedplanner.FederatedPlannerCostbased;
//import org.apache.sysds.test.AutomatedTestBase;
//import org.junit.Assert;
//import org.junit.Test;
//
//import java.util.ArrayList;
//import java.util.List;
//
//public class FTypeCombTest extends AutomatedTestBase {
//
//	@Override public void setUp() {}
//
//	@Test
//	public void ftypeCombTest(){
//		List<FType> secondInput = new ArrayList<>();
//		secondInput.add(null);
//		List<List<FType>> inputFTypes = List.of(
//			List.of(FType.ROW,FType.COL),
//			secondInput,
//			List.of(FType.BROADCAST,FType.FULL)
//		);
//
//		FederatedPlannerCostbased planner = new FederatedPlannerCostbased();
//		List<List<FType>> actualCombinations = planner.getAllCombinations(inputFTypes);
//
//		List<FType> expected1 = new ArrayList<>();
//		expected1.add(FType.ROW);
//		expected1.add(null);
//		expected1.add(FType.BROADCAST);
//		List<FType> expected2 = new ArrayList<>();
//		expected2.add(FType.ROW);
//		expected2.add(null);
//		expected2.add(FType.FULL);
//		List<FType> expected3 = new ArrayList<>();
//		expected3.add(FType.COL);
//		expected3.add(null);
//		expected3.add(FType.BROADCAST);
//		List<FType> expected4 = new ArrayList<>();
//		expected4.add(FType.COL);
//		expected4.add(null);
//		expected4.add(FType.FULL);
//		List<List<FType>> expectedCombinations = List.of(expected1,expected2, expected3, expected4);
//
//		Assert.assertEquals(expectedCombinations.size(), actualCombinations.size());
//		for (List<FType> expectedComb : expectedCombinations)
//			Assert.assertTrue(actualCombinations.contains(expectedComb));
//	}
//}
