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

package org.apache.sysds.test.component.resource;

import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.resource.CloudInstance;
import org.apache.sysds.resource.enumeration.EnumerationUtils.ConfigurationPoint;
import org.apache.sysds.resource.enumeration.EnumerationUtils.SolutionPoint;
import org.apache.sysds.resource.enumeration.Enumerator;
import org.apache.sysds.resource.enumeration.EnumerationUtils.InstanceSearchSpace;
import org.apache.sysds.resource.enumeration.InterestBasedEnumerator;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Assert;
import org.junit.Test;
import org.mockito.MockedStatic;
import org.mockito.Mockito;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;

import static org.apache.sysds.resource.CloudUtils.GBtoBytes;
import static org.apache.sysds.test.component.resource.ResourceTestUtils.*;
import static org.junit.Assert.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;

@net.jcip.annotations.NotThreadSafe
public class EnumeratorTests extends AutomatedTestBase {
	static {
		ConfigurationManager.getCompilerConfig().set(CompilerConfig.ConfigType.RESOURCE_OPTIMIZATION, true);
	}

	@Override
	public void setUp() {}

	@Test
	public void builderWithInstanceRangeTest() {
		// test the parsing of mechanism for instance family and instance size ranges
		HashMap<String, CloudInstance> availableInstances = getSimpleCloudInstanceMap();

		Enumerator defaultEnumerator = getGridBasedEnumeratorPrebuild().build();
		Assert.assertEquals(availableInstances.size(), defaultEnumerator.getInstances().size());

		Enumerator enumeratorWithInstanceRanges = getGridBasedEnumeratorPrebuild()
				.withInstanceFamilyRange(new String[]{"m5", "c5"})
				.withInstanceSizeRange(new String[]{"xlarge"})
				.build();
		List<CloudInstance> expectedInstancesList = availableInstances.values().stream()
				.filter(instance -> instance.getInstanceName().startsWith("m5.")
						|| instance.getInstanceName().startsWith("c5."))
				.filter(instance -> instance.getInstanceName().endsWith(".xlarge"))
				.collect(Collectors.toList());
		HashMap<String, CloudInstance> actualInstancesMap = enumeratorWithInstanceRanges.getInstances();
		for (CloudInstance expectedInstance : expectedInstancesList) {
			Assert.assertTrue(
					actualInstancesMap.containsKey(expectedInstance.getInstanceName())
			);
		}
	}

	@Test
	public void preprocessingGridBasedTest() {
		Enumerator gridBasedEnumerator = getGridBasedEnumeratorPrebuild().build();

		gridBasedEnumerator.preprocessing();
		// assertions for driver space
		InstanceSearchSpace driverSpace = gridBasedEnumerator.getDriverSpace();
		assertEquals(4, driverSpace.size());
		assertInstanceInSearchSpace("c5.xlarge", driverSpace, 8, 4, 0);
		assertInstanceInSearchSpace("c5d.xlarge", driverSpace, 8, 4, 1);
		assertInstanceInSearchSpace("c5n.xlarge", driverSpace, 10.5, 4, 0);
		assertInstanceInSearchSpace("m5.xlarge", driverSpace, 16, 4, 0);
		assertInstanceInSearchSpace("m5d.xlarge", driverSpace, 16, 4, 1);
		assertInstanceInSearchSpace("m5n.xlarge", driverSpace, 16, 4, 2);
		assertInstanceInSearchSpace("c5.2xlarge", driverSpace, 16, 8, 0);
		assertInstanceInSearchSpace("m5.2xlarge", driverSpace, 32, 8, 0);
		// assertions for executor space
		InstanceSearchSpace executorSpace = gridBasedEnumerator.getDriverSpace();
		assertEquals(4, executorSpace.size());
		assertInstanceInSearchSpace("c5.xlarge", executorSpace, 8, 4, 0);
		assertInstanceInSearchSpace("c5d.xlarge", executorSpace, 8, 4, 1);
		assertInstanceInSearchSpace("c5n.xlarge", executorSpace, 10.5, 4, 0);
		assertInstanceInSearchSpace("m5.xlarge", executorSpace, 16, 4, 0);
		assertInstanceInSearchSpace("m5d.xlarge", executorSpace, 16, 4, 1);
		assertInstanceInSearchSpace("m5n.xlarge", executorSpace, 16, 4, 2);
		assertInstanceInSearchSpace("c5.2xlarge", executorSpace, 16, 8, 0);
		assertInstanceInSearchSpace("m5.2xlarge", executorSpace, 32, 8, 0);
	}

	@Test
	public void preprocessingInterestBasedDriverMemoryTest() {
		Enumerator interestBasedEnumerator = getInterestBasedEnumeratorPrebuild()
				.withInterestEstimatesInCP(true)
				.withInterestBroadcastVars(false)
				.build();

		// use 10GB (scaled) memory estimate to be between the available 8GB and 16GB driver node's memory
		TreeSet<Long> mockingMemoryEstimates = new TreeSet<>(Set.of(GBtoBytes(10)));
		try (MockedStatic<InterestBasedEnumerator> mockedEnumerator =
					 Mockito.mockStatic(InterestBasedEnumerator.class, Mockito.CALLS_REAL_METHODS)) {
			mockedEnumerator
					.when(() -> InterestBasedEnumerator.getMemoryEstimates(
							any(Program.class),
							eq(false),
							eq(OptimizerUtils.MEM_UTIL_FACTOR)))
					.thenReturn(mockingMemoryEstimates);
			interestBasedEnumerator.preprocessing();
		}

		// assertions for driver space
		InstanceSearchSpace driverSpace = interestBasedEnumerator.getDriverSpace();
		assertEquals(1, driverSpace.size());
		assertInstanceInSearchSpace("c5.xlarge", driverSpace, 8, 4, 0);
		Assert.assertNull(driverSpace.get(GBtoBytes(32)));
		// assertions for executor space
		InstanceSearchSpace executorSpace = interestBasedEnumerator.getExecutorSpace();
		assertEquals(4, executorSpace.size());
		assertInstanceInSearchSpace("c5.xlarge", executorSpace, 8, 4, 0);
		assertInstanceInSearchSpace("c5d.xlarge", executorSpace, 8, 4, 1);
		assertInstanceInSearchSpace("c5n.xlarge", executorSpace, 10.5, 4, 0);
		assertInstanceInSearchSpace("m5.xlarge", executorSpace, 16, 4, 0);
		assertInstanceInSearchSpace("m5d.xlarge", executorSpace, 16, 4, 1);
		assertInstanceInSearchSpace("m5n.xlarge", executorSpace, 16, 4, 2);
		assertInstanceInSearchSpace("c5.2xlarge", executorSpace, 16, 8, 0);
		assertInstanceInSearchSpace("m5.2xlarge", executorSpace, 32, 8, 0);
	}

	@Test
	public void preprocessingInterestBasedBroadcastMemoryTest() {
		Enumerator interestBasedEnumerator = getInterestBasedEnumeratorPrebuild()
				.withInterestEstimatesInCP(false)
				.withInterestBroadcastVars(true)
				.build();

		double outputEstimate = 2.5;
		double scaledOutputEstimateBroadcast = outputEstimate / InterestBasedEnumerator.BROADCAST_MEMORY_FACTOR; // ~=12
		// scaledOutputEstimateCP = 2 * outputEstimate / OptimizerUtils.MEM_UTIL_FACTOR ~= 7
		TreeSet<Long> mockingMemoryEstimates = new TreeSet<>(Set.of(GBtoBytes(scaledOutputEstimateBroadcast)));
		try (MockedStatic<InterestBasedEnumerator> mockedEnumerator =
					 Mockito.mockStatic(InterestBasedEnumerator.class, Mockito.CALLS_REAL_METHODS)) {
			mockedEnumerator
					.when(() -> InterestBasedEnumerator.getMemoryEstimates(
							any(Program.class),
							eq(true),
							eq(InterestBasedEnumerator.BROADCAST_MEMORY_FACTOR)))
					.thenReturn(mockingMemoryEstimates);
			interestBasedEnumerator.preprocessing();
		}

		// assertions for driver space
		InstanceSearchSpace driverSpace = interestBasedEnumerator.getDriverSpace();
		assertEquals(1, driverSpace.size());
		assertInstanceInSearchSpace("c5.xlarge", driverSpace, 8, 4, 0);
		Assert.assertNull(driverSpace.get(GBtoBytes(16)));
		Assert.assertNull(driverSpace.get(GBtoBytes(32)));
		// assertions for executor space
		InstanceSearchSpace executorSpace = interestBasedEnumerator.getExecutorSpace();
		assertEquals(2, executorSpace.size());
		assertInstanceInSearchSpace("m5.xlarge", executorSpace, 16, 4, 0);
		assertInstanceInSearchSpace("m5d.xlarge", executorSpace, 16, 4, 1);
		assertInstanceInSearchSpace("m5n.xlarge", executorSpace, 16, 4, 2);
		assertInstanceInSearchSpace("m5.2xlarge", executorSpace, 32, 8, 0);
		Assert.assertNull(executorSpace.get(GBtoBytes(10.5)));
	}

	@Test
	public void updateOptimalSolutionMinCostsTest() {
		ConfigurationPoint dummyConfig = new ConfigurationPoint(null, null, -1);
		SolutionPoint currentSolution;
		Program emptyProgram = new Program();
		HashMap<String, CloudInstance> instances = ResourceTestUtils.getSimpleCloudInstanceMap();
		Enumerator enumerator = (new Enumerator.Builder())
				.withRuntimeProgram(emptyProgram)
				.withAvailableInstances(instances)
				.withEnumerationStrategy(Enumerator.EnumerationStrategy.GridBased)
				.withOptimizationStrategy(Enumerator.OptimizationStrategy.MinCosts)
				.build();

		currentSolution = enumerator.getOptimalSolution();
		Assert.assertEquals(Double.MAX_VALUE, currentSolution.getTimeCost(), 0);
		Assert.assertEquals(Double.MAX_VALUE, currentSolution.getMonetaryCost(), 0);

		enumerator.updateOptimalSolution(100, 100, dummyConfig);
		currentSolution = enumerator.getOptimalSolution();
		Assert.assertEquals(100, currentSolution.getTimeCost(), 0);
		Assert.assertEquals(100, currentSolution.getMonetaryCost(), 0);

		enumerator.updateOptimalSolution(90, 1000, dummyConfig);
		currentSolution = enumerator.getOptimalSolution();
		Assert.assertEquals(100, currentSolution.getTimeCost(), 0);
		Assert.assertEquals(100, currentSolution.getMonetaryCost(), 0);

		enumerator.updateOptimalSolution(200, 99, dummyConfig);
		currentSolution = enumerator.getOptimalSolution();
		Assert.assertEquals(100, currentSolution.getTimeCost(), 0);
		Assert.assertEquals(100, currentSolution.getMonetaryCost(), 0);

		enumerator.updateOptimalSolution(101, 99, dummyConfig);
		currentSolution = enumerator.getOptimalSolution();
		Assert.assertEquals(101, currentSolution.getTimeCost(), 0);
		Assert.assertEquals(99, currentSolution.getMonetaryCost(), 0);

		enumerator.updateOptimalSolution(99, 100, dummyConfig);
		currentSolution = enumerator.getOptimalSolution();
		Assert.assertEquals(101, currentSolution.getTimeCost(), 0);
		Assert.assertEquals(99, currentSolution.getMonetaryCost(), 0);

		enumerator.updateOptimalSolution(0.5, 100, dummyConfig);
		currentSolution = enumerator.getOptimalSolution();
		Assert.assertEquals(0.5, currentSolution.getTimeCost(), 0);
		Assert.assertEquals(100, currentSolution.getMonetaryCost(), 0);
	}

	@Test
	public void updateOptimalSolutionMinTimeTest() {
		ConfigurationPoint dummyConfig = new ConfigurationPoint(null, null, -1);
		SolutionPoint currentSolution;
		Program emptyProgram = new Program();
		HashMap<String, CloudInstance> instances = ResourceTestUtils.getSimpleCloudInstanceMap();
		Enumerator.setMinPrice(100);
		Enumerator enumerator = (new Enumerator.Builder())
				.withRuntimeProgram(emptyProgram)
				.withAvailableInstances(instances)
				.withEnumerationStrategy(Enumerator.EnumerationStrategy.GridBased)
				.withOptimizationStrategy(Enumerator.OptimizationStrategy.MinTime)
				.build();


		currentSolution = enumerator.getOptimalSolution();
		Assert.assertEquals(Double.MAX_VALUE, currentSolution.getTimeCost(), 0);
		Assert.assertEquals(Double.MAX_VALUE, currentSolution.getMonetaryCost(), 0);

		enumerator.updateOptimalSolution(100, 101, dummyConfig);
		currentSolution = enumerator.getOptimalSolution();
		Assert.assertEquals(Double.MAX_VALUE, currentSolution.getTimeCost(), 0);
		Assert.assertEquals(Double.MAX_VALUE, currentSolution.getMonetaryCost(), 0);

		enumerator.updateOptimalSolution(90, 100, dummyConfig);
		currentSolution = enumerator.getOptimalSolution();
		Assert.assertEquals(90, currentSolution.getTimeCost(), 0);
		Assert.assertEquals(100, currentSolution.getMonetaryCost(), 0);

		enumerator.updateOptimalSolution(80, 10, dummyConfig);
		currentSolution = enumerator.getOptimalSolution();
		Assert.assertEquals(80, currentSolution.getTimeCost(), 0);
		Assert.assertEquals(10, currentSolution.getMonetaryCost(), 0);

		enumerator.updateOptimalSolution(10, 100, dummyConfig);
		currentSolution = enumerator.getOptimalSolution();
		Assert.assertEquals(10, currentSolution.getTimeCost(), 0);
		Assert.assertEquals(100, currentSolution.getMonetaryCost(), 0);
	}

	@Test
	public void updateOptimalSolutionMinPriceTest() {
		ConfigurationPoint dummyConfig = new ConfigurationPoint(null, null, -1);
		SolutionPoint currentSolution;
		Program emptyProgram = new Program();
		HashMap<String, CloudInstance> instances = ResourceTestUtils.getSimpleCloudInstanceMap();
		Enumerator.setMinTime(600);
		Enumerator enumerator = (new Enumerator.Builder())
				.withRuntimeProgram(emptyProgram)
				.withAvailableInstances(instances)
				.withEnumerationStrategy(Enumerator.EnumerationStrategy.GridBased)
				.withOptimizationStrategy(Enumerator.OptimizationStrategy.MinPrice)
				.build();

		currentSolution = enumerator.getOptimalSolution();
		Assert.assertEquals(Double.MAX_VALUE, currentSolution.getTimeCost(), 0);
		Assert.assertEquals(Double.MAX_VALUE, currentSolution.getMonetaryCost(), 0);

		enumerator.updateOptimalSolution(601, 100, dummyConfig);
		currentSolution = enumerator.getOptimalSolution();
		Assert.assertEquals(Double.MAX_VALUE, currentSolution.getTimeCost(), 0);
		Assert.assertEquals(Double.MAX_VALUE, currentSolution.getMonetaryCost(), 0);

		enumerator.updateOptimalSolution(100, 90, dummyConfig);
		currentSolution = enumerator.getOptimalSolution();
		Assert.assertEquals(100, currentSolution.getTimeCost(), 0);
		Assert.assertEquals(90, currentSolution.getMonetaryCost(), 0);

		enumerator.updateOptimalSolution(10, 80, dummyConfig);
		currentSolution = enumerator.getOptimalSolution();
		Assert.assertEquals(10, currentSolution.getTimeCost(), 0);
		Assert.assertEquals(80, currentSolution.getMonetaryCost(), 0);

		enumerator.updateOptimalSolution(100, 10, dummyConfig);
		currentSolution = enumerator.getOptimalSolution();
		Assert.assertEquals(100, currentSolution.getTimeCost(), 0);
		Assert.assertEquals(10, currentSolution.getMonetaryCost(), 0);
	}

	@Test
	public void evaluateSingleNodeExecutionGridBasedTest() {
		Enumerator gridBasedEnumerator;
		boolean result;

		gridBasedEnumerator = getGridBasedEnumeratorPrebuild()
				.withNumberExecutorsRange(0,1)
				.build();

		// memory not relevant for grid-based enumerator
		result = gridBasedEnumerator.evaluateSingleNodeExecution(-1, 1);
		Assert.assertTrue(result);

		gridBasedEnumerator = getGridBasedEnumeratorPrebuild()
				.withNumberExecutorsRange(1,2)
				.build();

		// memory not relevant for grid-based enumerator
		result = gridBasedEnumerator.evaluateSingleNodeExecution(-1, 1);
		Assert.assertFalse(result);
	}

	@Test
	public void estimateRangeExecutorsGridBasedStepSizeTest() {
		Enumerator gridBasedEnumerator;

		// num. executors range starting from zero and step size = 2
		gridBasedEnumerator = getGridBasedEnumeratorPrebuild()
				.withNumberExecutorsRange(0, 10)
				.withStepSizeExecutor(2)
				.build();
		// test the general case when the max level of parallelism is not reached (0 is never part of the result)
		List<Integer> expectedResult = new ArrayList<>(List.of(2, 4, 6, 8, 10));
		List<Integer> actualResult = gridBasedEnumerator.estimateRangeExecutors(1, -1, 4);
		Assert.assertEquals(expectedResult, actualResult);
		// test the case when the max level of parallelism (1000) is reached (0 is never part of the result)
		expectedResult = new ArrayList<>(List.of(2, 4));
		actualResult = gridBasedEnumerator.estimateRangeExecutors(1, -1, 200);
		Assert.assertEquals(expectedResult, actualResult);

		// num. executors range not starting from zero and without step size given
		gridBasedEnumerator = getGridBasedEnumeratorPrebuild()
				.withNumberExecutorsRange(3, 8)
				.build();
		// test the general case when the max level of parallelism is not reached (0 is never part of the result)
		expectedResult = new ArrayList<>(List.of(3, 4, 5, 6, 7, 8));
		actualResult = gridBasedEnumerator.estimateRangeExecutors(1, -1, 4);
		Assert.assertEquals(expectedResult, actualResult);
		// test the case when the max level of parallelism (1000) is reached (0 is never part of the result)
		expectedResult = new ArrayList<>(List.of(3, 4, 5));
		actualResult = gridBasedEnumerator.estimateRangeExecutors(1, -1, 200);
		Assert.assertEquals(expectedResult, actualResult);
	}

	@Test
	public void estimateRangeExecutorsGridBasedExpBaseTest() {
		Enumerator gridBasedEnumerator;
		ArrayList<Integer> expectedResult;
		ArrayList<Integer> actualResult;

		// num. executors range starting from zero and exponential base = 2
		gridBasedEnumerator = getGridBasedEnumeratorPrebuild()
				.withNumberExecutorsRange(0, 10)
				.withExpBaseExecutors(2)
				.build();
		// test the general case when the max level of parallelism is not reached (0 is never part of the result)
		expectedResult = new ArrayList<>(List.of(1, 2, 4, 8));
		actualResult = gridBasedEnumerator.estimateRangeExecutors(1, -1, 4);
		Assert.assertEquals(expectedResult, actualResult);
		// test the case when the max level of parallelism (1000) is reached (0 is never part of the result)
		expectedResult = new ArrayList<>(List.of(1, 2, 4));
		actualResult = gridBasedEnumerator.estimateRangeExecutors(1, -1, 200);
		Assert.assertEquals(expectedResult, actualResult);

		// num. executors range not starting from zero and with exponential base = 3
		gridBasedEnumerator = getGridBasedEnumeratorPrebuild()
				.withNumberExecutorsRange(3, 30)
				.withExpBaseExecutors(3)
				.build();
		// test the general case when the max level of parallelism is not reached (0 is never part of the result)
		expectedResult = new ArrayList<>(List.of(3,9, 27));
		actualResult = gridBasedEnumerator.estimateRangeExecutors(1, -1, 4);
		Assert.assertEquals(expectedResult, actualResult);
		// test the case when the max level of parallelism (1000) is reached (0 is never part of the result)
		expectedResult = new ArrayList<>(List.of(3,9));
		actualResult = gridBasedEnumerator.estimateRangeExecutors(1, -1, 100);
		Assert.assertEquals(expectedResult, actualResult);
	}

	@Test
	public void evaluateSingleNodeExecutionInterestBasedTest() {
		boolean result;

		// no fitting the memory estimates for caching
		Enumerator interestBasedEnumerator = getInterestBasedEnumeratorPrebuild()
				.withNumberExecutorsRange(0, 5)
				.withInterestEstimatesInCP(false)
				.withInterestBroadcastVars(false)
				.withInterestLargestEstimate(true)
				.build();

		TreeSet<Long> mockingMemoryEstimates = new TreeSet<>(Set.of(GBtoBytes(6), GBtoBytes(12)));
		try (MockedStatic<InterestBasedEnumerator> mockedEnumerator =
					 Mockito.mockStatic(InterestBasedEnumerator.class, Mockito.CALLS_REAL_METHODS)) {
			mockedEnumerator
					.when(() -> InterestBasedEnumerator.getMemoryEstimates(
							any(Program.class),
							eq(false),
							eq(InterestBasedEnumerator.MEMORY_FACTOR)))
					.thenReturn(mockingMemoryEstimates);
			// initiate memoryEstimatesSpark
			interestBasedEnumerator.preprocessing();
		}

		result = interestBasedEnumerator.evaluateSingleNodeExecution(GBtoBytes(8), 1);
		Assert.assertFalse(result);
	}

	@Test
	public void estimateRangeExecutorsInterestBasedAllEnabledTest() {
		ArrayList<Integer> expectedResult;
		ArrayList<Integer>actualResult;

		// no fitting the memory estimates for checkpointing
		Enumerator interestBasedEnumerator = getInterestBasedEnumeratorPrebuild()
				.withNumberExecutorsRange(0, 5)
				.withInterestOutputCaching(true)
				.build();
		interestBasedEnumerator.preprocessing();
		// test the general case of limiting to only one executor for the empty program (no memory estimates)
		expectedResult = new ArrayList<>(List.of(1));
		actualResult = interestBasedEnumerator.estimateRangeExecutors(1, -1, 100);
		Assert.assertEquals(expectedResult, actualResult);
	}

	@Test
	public void estimateRangeExecutorsInterestBasedNoInterestOutputCachingTest() {
		ArrayList<Integer> expectedResult;
		ArrayList<Integer>actualResult;

		// no fitting the memory estimates for checkpointing
		Enumerator interestBasedEnumerator = getInterestBasedEnumeratorPrebuild()
				.withNumberExecutorsRange(0, 5)
				.withInterestOutputCaching(false) // explicit but also default
				.build();
		interestBasedEnumerator.preprocessing();
		// test the general case when the max level of parallelism is not reached (0 is never part of the result)
		expectedResult = new ArrayList<>(List.of(1, 2, 3, 4, 5));
		actualResult = interestBasedEnumerator.estimateRangeExecutors(1, -1, 4);
		Assert.assertEquals(expectedResult, actualResult);
		// test the case when the max level of parallelism (1152) is reached (0 is never part of the result)
		expectedResult = new ArrayList<>(List.of(1, 2, 3, 4));
		actualResult = interestBasedEnumerator.estimateRangeExecutors(1, -1, 256);
		Assert.assertEquals(expectedResult, actualResult);
	}

	@Test
	public void estimateRangeExecutorsInterestBasedCheckpointMemoryTest() {
		// fitting the memory estimates for checkpointing
		Enumerator interestBasedEnumerator = getInterestBasedEnumeratorPrebuild()
				.withNumberExecutorsRange(0, 5)
				.withInterestEstimatesInCP(false)
				.withInterestBroadcastVars(false)
				.withInterestOutputCaching(true)
				.build();

		TreeSet<Long> mockingMemoryEstimates = new TreeSet<>(Set.of(GBtoBytes(20), GBtoBytes(40)));
		try (MockedStatic<InterestBasedEnumerator> mockedEnumerator =
					 Mockito.mockStatic(InterestBasedEnumerator.class, Mockito.CALLS_REAL_METHODS)) {
			mockedEnumerator
					.when(() -> InterestBasedEnumerator.getMemoryEstimates(
							any(Program.class),
							eq(true),
							eq(InterestBasedEnumerator.CACHE_MEMORY_FACTOR)))
					.thenReturn(mockingMemoryEstimates);
			// initiate memoryEstimatesSpark
			interestBasedEnumerator.preprocessing();
		}

		// test the general case when the max level of parallelism is not reached (0 is never part of the result)
		List<Integer> expectedResult = new ArrayList<>(List.of(1, 2, 3));
		List<Integer> actualResult = interestBasedEnumerator.estimateRangeExecutors(1, GBtoBytes(16), 4);
		Assert.assertEquals(expectedResult, actualResult);
		// test the case when the max level of parallelism (1000) is reached (0 is never part of the result)
		expectedResult = new ArrayList<>(List.of(1, 2));
		actualResult = interestBasedEnumerator.estimateRangeExecutors(1, GBtoBytes(16), 500);
		Assert.assertEquals(expectedResult, actualResult);
	}

	@Test
	public void processingTest() {
		// all implemented enumerators should enumerate the same solution pool in this basic case - empty program
		Enumerator gridBasedEnumerator = getGridBasedEnumeratorPrebuild()
				.withNumberExecutorsRange(0, 2)
				.build();

		Enumerator interestBasedEnumerator = getInterestBasedEnumeratorPrebuild()
				.withNumberExecutorsRange(0, 2)
				.build();

		HashMap<String, CloudInstance> instances = ResourceTestUtils.getSimpleCloudInstanceMap();
		InstanceSearchSpace space = new InstanceSearchSpace();
		space.initSpace(instances);

		// run processing for the grid based enumerator
		gridBasedEnumerator.setDriverSpace(space);
		gridBasedEnumerator.setExecutorSpace(space);
		gridBasedEnumerator.preprocessing();
		gridBasedEnumerator.processing();
		SolutionPoint actualSolutionGB = gridBasedEnumerator.postprocessing();
		// run processing for the interest based enumerator
		interestBasedEnumerator.setDriverSpace(space);
		interestBasedEnumerator.setExecutorSpace(space);
		interestBasedEnumerator.preprocessing();
		interestBasedEnumerator.processing();
		SolutionPoint actualSolutionIB = gridBasedEnumerator.postprocessing();

		// expected solution with 0 executors (number executors = 0, executors and executorInstance being null)
		// and the cheapest instance for the driver
		// Grid-Based
		Assert.assertEquals(0, actualSolutionGB.numberExecutors);
		assertEqualsCloudInstances(instances.get("c5.xlarge"), actualSolutionGB.driverInstance);
		Assert.assertNull(actualSolutionIB.executorInstance);
		// Interest-Based
		Assert.assertEquals(0, actualSolutionIB.numberExecutors);
		assertEqualsCloudInstances(instances.get("c5.xlarge"), actualSolutionIB.driverInstance);
		Assert.assertNull(actualSolutionIB.executorInstance);
	}

	@Test
	public void GridBasedEnumerationMinPriceTest() {
		Enumerator gridBasedEnumerator = getGridBasedEnumeratorPrebuild()
				.withNumberExecutorsRange(0, 2)
				.build();

		gridBasedEnumerator.preprocessing();
		gridBasedEnumerator.processing();
		SolutionPoint solution = gridBasedEnumerator.postprocessing();

		// expected c5.xlarge since it is the cheaper
		Assert.assertEquals("c5.xlarge", solution.driverInstance.getInstanceName());
		// expected no executor nodes since tested for a 'zero' program
		Assert.assertEquals(0, solution.numberExecutors);
	}

	@Test
	public void InterestBasedEnumerationMinPriceTest() {
		Enumerator interestBasedEnumerator = getInterestBasedEnumeratorPrebuild()
				.withNumberExecutorsRange(0, 2)
				.build();

		interestBasedEnumerator.preprocessing();
		interestBasedEnumerator.processing();
		SolutionPoint solution = interestBasedEnumerator.postprocessing();

		// expected c5.xlarge since is the instance with at least memory
		Assert.assertEquals("c5.xlarge", solution.driverInstance.getInstanceName());
		// expected no executor nodes since tested for a 'zero' program
		Assert.assertEquals(0, solution.numberExecutors);
	}

	@Test
	public void GridBasedEnumerationMinTimeTest() {
		Enumerator gridBasedEnumerator = getGridBasedEnumeratorPrebuild()
				.withOptimizationStrategy(Enumerator.OptimizationStrategy.MinPrice)
				.withNumberExecutorsRange(0, 2)
				.build();

		gridBasedEnumerator.preprocessing();
		gridBasedEnumerator.processing();
		SolutionPoint solution = gridBasedEnumerator.postprocessing();

		// expected c5.xlarge since it is the cheaper
		Assert.assertEquals("c5.xlarge", solution.driverInstance.getInstanceName());
		// expected no executor nodes since tested for a 'zero' program
		Assert.assertEquals(0, solution.numberExecutors);
	}

	@Test
	public void InterestBasedEnumerationMinTimeTest() {
		Enumerator interestBasedEnumerator = getInterestBasedEnumeratorPrebuild()
				.withOptimizationStrategy(Enumerator.OptimizationStrategy.MinTime)
				.withNumberExecutorsRange(0, 2)
				.build();

		interestBasedEnumerator.preprocessing();
		interestBasedEnumerator.processing();
		SolutionPoint solution = interestBasedEnumerator.postprocessing();

		// expected c5.xlarge since is the instance with at least memory
		Assert.assertEquals("c5.xlarge", solution.driverInstance.getInstanceName());
		// expected no executor nodes since tested for a 'zero' program
		Assert.assertEquals(0, solution.numberExecutors);
	}

	@Test
	public void PruneBasedEnumerationMinTimeTest() {
		Enumerator pruneBasedEnumerator = getPruneBasedEnumeratorPrebuild()
				.withNumberExecutorsRange(0, 2)
				.build();

		pruneBasedEnumerator.preprocessing();
		pruneBasedEnumerator.processing();
		SolutionPoint solution = pruneBasedEnumerator.postprocessing();

		// expected c5.xlarge since it is the cheaper
		Assert.assertEquals("c5.xlarge", solution.driverInstance.getInstanceName());
		// expected no executor nodes since tested for a 'zero' program
		Assert.assertEquals(0, solution.numberExecutors);
	}


	// Helpers ---------------------------------------------------------------------------------------------------------

	private static Enumerator.Builder getGridBasedEnumeratorPrebuild() {
		Program emptyProgram = new Program();
		HashMap<String, CloudInstance> instances = ResourceTestUtils.getSimpleCloudInstanceMap();
		return (new Enumerator.Builder())
				.withRuntimeProgram(emptyProgram)
				.withAvailableInstances(instances)
				.withEnumerationStrategy(Enumerator.EnumerationStrategy.GridBased)
				.withOptimizationStrategy(Enumerator.OptimizationStrategy.MinPrice);
	}

	private static Enumerator.Builder getInterestBasedEnumeratorPrebuild() {
		Program emptyProgram = new Program();
		HashMap<String, CloudInstance> instances = ResourceTestUtils.getSimpleCloudInstanceMap();
		return (new Enumerator.Builder())
				.withRuntimeProgram(emptyProgram)
				.withAvailableInstances(instances)
				.withEnumerationStrategy(Enumerator.EnumerationStrategy.InterestBased)
				.withOptimizationStrategy(Enumerator.OptimizationStrategy.MinPrice);
	}

	private static Enumerator.Builder getPruneBasedEnumeratorPrebuild() {
		Program emptyProgram = new Program();
		HashMap<String, CloudInstance> instances = ResourceTestUtils.getSimpleCloudInstanceMap();
		return (new Enumerator.Builder())
				.withRuntimeProgram(emptyProgram)
				.withAvailableInstances(instances)
				.withEnumerationStrategy(Enumerator.EnumerationStrategy.PruneBased)
				.withOptimizationStrategy(Enumerator.OptimizationStrategy.MinPrice);
	}

	private static void assertInstanceInSearchSpace(
			String expectedName,
			InstanceSearchSpace searchSpace,
			double memory, /* in GB */
			int cores,
			int index
	) {
		Assert.assertNotNull(searchSpace.get(GBtoBytes(memory)));
		try {
			String actualName = searchSpace.get(GBtoBytes(memory)).get(cores).get(index).getInstanceName();
			Assert.assertEquals(expectedName, actualName);
		} catch (NullPointerException e) {
			fail(expectedName+" instances not properly passed to "+searchSpace.getClass().getName());
		}
	}
}
