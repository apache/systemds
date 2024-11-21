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

package org.apache.sysds.test.component.federated;

import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.fedplanner.FTypes;
import org.apache.sysds.hops.fedplanner.MemoTable;
import org.apache.sysds.hops.fedplanner.MemoTable.FedPlan;
import org.apache.commons.lang3.tuple.Pair;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.when;

public class MemoTableTest {
	
	private MemoTable memoTable;
	
	@Mock
	private Hop mockHop1;
	
	@Mock
	private Hop mockHop2;
	
	private java.util.Random rand;

	@Before
	public void setUp() {
		MockitoAnnotations.openMocks(this);
		memoTable = new MemoTable();
		
		// Set up unique IDs for mock Hops
		when(mockHop1.getHopID()).thenReturn(1L);
		when(mockHop2.getHopID()).thenReturn(2L);
		
		// Initialize random generator with fixed seed for reproducible tests
		rand = new java.util.Random(42); 
	}
	
	@Test
	public void testAddAndGetSingleFedPlan() {
		// Initialize test data
		List<Pair<Long, FTypes.FType>> planRefs = new ArrayList<>();
		FedPlan fedPlan = new FedPlan(mockHop1, 10.0, planRefs);
		
		// Verify initial state
		List<FedPlan> result = memoTable.get(mockHop1, FTypes.FType.FULL);
		assertNull("Initial FedPlan list should be null before adding any plans", result);

		// Add single FedPlan
		memoTable.addFedPlan(mockHop1, FTypes.FType.FULL, fedPlan);
		
		// Verify after addition
		result = memoTable.get(mockHop1, FTypes.FType.FULL);
		assertNotNull("FedPlan list should exist after adding a plan", result);
		assertEquals("FedPlan list should contain exactly one plan", 1, result.size());
		assertEquals("FedPlan cost should be exactly 10.0", 10.0, result.get(0).getCost(), 0.001);
	}
	
	@Test
	public void testAddMultipleDuplicatedFedPlans() {
		// Initialize test data with duplicate costs
		List<Pair<Long, FTypes.FType>> planRefs = new ArrayList<>();
		List<FedPlan> fedPlans = new ArrayList<>();
		fedPlans.add(new FedPlan(mockHop1, 10.0, planRefs));  // Unique cost
		fedPlans.add(new FedPlan(mockHop1, 20.0, planRefs));  // First duplicate
		fedPlans.add(new FedPlan(mockHop1, 20.0, planRefs));  // Second duplicate
		
		// Add multiple plans including duplicates
		memoTable.addFedPlanList(mockHop1, FTypes.FType.FULL, fedPlans);
		
		// Verify handling of duplicate plans
		List<FedPlan> result = memoTable.get(mockHop1, FTypes.FType.FULL);
		assertNotNull("FedPlan list should exist after adding multiple plans", result);
		assertEquals("FedPlan list should maintain all plans including duplicates", 3, result.size());
	}
	
	@Test
	public void testContains() {
		// Initialize test data
		List<Pair<Long, FTypes.FType>> planRefs = new ArrayList<>();
		FedPlan fedPlan = new FedPlan(mockHop1, 10.0, planRefs);
		
		// Verify initial state
		assertFalse("MemoTable should not contain any entries initially", 
			memoTable.contains(mockHop1, FTypes.FType.FULL));
		
		// Add plan and verify presence
		memoTable.addFedPlan(mockHop1, FTypes.FType.FULL, fedPlan);
		
		assertTrue("MemoTable should contain entry after adding FedPlan", 
			memoTable.contains(mockHop1, FTypes.FType.FULL));
		assertFalse("MemoTable should not contain entries for different Hop", 
			memoTable.contains(mockHop2, FTypes.FType.FULL));
	}
	
	@Test
	public void testPrunePlanPruneAll() {
		// Initialize base test data
		List<Pair<Long, FTypes.FType>> planRefs = new ArrayList<>();
		// Create separate FedPlan lists for independent testing of each Hop
		List<FedPlan> fedPlans1 = new ArrayList<>();  // Plans for mockHop1
		List<FedPlan> fedPlans2 = new ArrayList<>();  // Plans for mockHop2
		
		// Generate random cost FedPlans for both Hops
		double minCost = Double.MAX_VALUE;
		int size = 100;
		for(int i = 0; i < size; i++) {
			double cost = rand.nextDouble() * 1000;  // Random cost between 0 and 1000
			fedPlans1.add(new FedPlan(mockHop1, cost, planRefs));
			fedPlans2.add(new FedPlan(mockHop2, cost, planRefs));
			minCost = Math.min(minCost, cost);
		}
		
		// Add FedPlan lists to MemoTable
		memoTable.addFedPlanList(mockHop1, FTypes.FType.FULL, fedPlans1);
		memoTable.addFedPlanList(mockHop2, FTypes.FType.FULL, fedPlans2);
		
		// Test selective pruning on mockHop1
		memoTable.prunePlan(mockHop1, FTypes.FType.FULL);
		
		// Get results for verification
		List<FedPlan> result1 = memoTable.get(mockHop1, FTypes.FType.FULL);
		List<FedPlan> result2 = memoTable.get(mockHop2, FTypes.FType.FULL);

		// Verify selective pruning results
		assertNotNull("Pruned mockHop1 should maintain a FedPlan list", result1);
		assertEquals("Pruned mockHop1 should contain exactly one minimum cost plan", 1, result1.size());
		assertEquals("Pruned mockHop1's plan should have the minimum cost", minCost, result1.get(0).getCost(), 0.001);
		
		// Verify unpruned Hop state
		assertNotNull("Unpruned mockHop2 should maintain a FedPlan list", result2);
		assertEquals("Unpruned mockHop2 should maintain all original plans", size, result2.size());

		// Add additional plans to both Hops
		for(int i = 0; i < size; i++) {
			double cost = rand.nextDouble() * 1000;
			memoTable.addFedPlan(mockHop1, FTypes.FType.FULL, new FedPlan(mockHop1, cost, planRefs));
			memoTable.addFedPlan(mockHop2, FTypes.FType.FULL, new FedPlan(mockHop2, cost, planRefs));
			minCost = Math.min(minCost, cost);
		}

		// Test global pruning
		memoTable.pruneAll();
		
		// Verify global pruning results
		assertNotNull("mockHop1 should maintain a FedPlan list after global pruning", result1);
		assertEquals("mockHop1 should contain exactly one minimum cost plan after global pruning", 
			1, result1.size());
		assertEquals("mockHop1's plan should have the global minimum cost", 
			minCost, result1.get(0).getCost(), 0.001);

		assertNotNull("mockHop2 should maintain a FedPlan list after global pruning", result2);
		assertEquals("mockHop2 should contain exactly one minimum cost plan after global pruning", 
			1, result2.size());
		assertEquals("mockHop2's plan should have the global minimum cost", 
			minCost, result2.get(0).getCost(), 0.001);
	}
}
