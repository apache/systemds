package org.apache.sysds.test.component.resource;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.resource.CloudInstance;
import org.apache.sysds.resource.enumeration.Enumerator;
import org.apache.sysds.resource.enumeration.EnumerationUtils.InstanceSearchSpace;
import org.apache.sysds.resource.enumeration.EnumerationUtils.ConfigurationPoint;
import org.apache.sysds.resource.enumeration.EnumerationUtils.SolutionPoint;
import org.apache.sysds.resource.enumeration.InterestBasedEnumerator;
import org.apache.sysds.runtime.controlprogram.Program;
import org.junit.Assert;
import org.junit.Test;
import org.mockito.MockedStatic;
import org.mockito.Mockito;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;

import static org.apache.sysds.resource.CloudUtils.GBtoBytes;
import static org.junit.Assert.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;

public class EnumeratorTests {

    @Test
    public void loadInstanceTableTest() throws IOException {
        // loading the table is entirely implemented by the abstract class
        // use any enumerator
        Enumerator anyEnumerator = getGridBasedEnumeratorPrebuild()
                .withInstanceTypeRange(new String[]{"m5"})
                .withInstanceSizeRange(new String[]{"xlarge"})
                .build();

        File tmpFile = TestingUtils.generateTmpInstanceInfoTableFile();
        anyEnumerator.loadInstanceTableFile(tmpFile.toString());

        HashMap<String, CloudInstance> actualInstances = anyEnumerator.getInstances();

        Assert.assertEquals(1, actualInstances.size());
        Assert.assertNotNull(actualInstances.get("m5.xlarge"));

        Files.deleteIfExists(tmpFile.toPath());
    }

    @Test
    public void preprocessingGridBasedTest() {
        Enumerator gridBasedEnumerator = getGridBasedEnumeratorPrebuild().build();

        HashMap<String, CloudInstance> instances = TestingUtils.getSimpleCloudInstanceMap();
        gridBasedEnumerator.setInstanceTable(instances);

        gridBasedEnumerator.preprocessing();
        // assertions for driver space
        InstanceSearchSpace driverSpace = gridBasedEnumerator.getDriverSpace();
        assertEquals(3, driverSpace.size());
        assertInstanceInSearchSpace("c5.xlarge", driverSpace, 8, 4, 0);
        assertInstanceInSearchSpace("m5.xlarge", driverSpace, 16, 4, 0);
        assertInstanceInSearchSpace("c5.2xlarge", driverSpace, 16, 8, 0);
        assertInstanceInSearchSpace("m5.2xlarge", driverSpace, 32, 8, 0);
        // assertions for executor space
        InstanceSearchSpace executorSpace = gridBasedEnumerator.getDriverSpace();
        assertEquals(3, executorSpace.size());
        assertInstanceInSearchSpace("c5.xlarge", executorSpace, 8, 4, 0);
        assertInstanceInSearchSpace("m5.xlarge", executorSpace, 16, 4, 0);
        assertInstanceInSearchSpace("c5.2xlarge", executorSpace, 16, 8, 0);
        assertInstanceInSearchSpace("m5.2xlarge", executorSpace, 32, 8, 0);
    }

    @Test
    public void preprocessingInterestBasedDriverMemoryTest() {
        Enumerator interestBasedEnumerator = getInterestBasedEnumeratorPrebuild()
                .withFitDriverMemory(true)
                .withFitBroadcastMemory(false)
                .build();

        HashMap<String, CloudInstance> instances = TestingUtils.getSimpleCloudInstanceMap();
        interestBasedEnumerator.setInstanceTable(instances);

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
        assertEquals(2, driverSpace.size());
        assertInstanceInSearchSpace("c5.xlarge", driverSpace, 8, 4, 0);
        assertInstanceInSearchSpace("m5.xlarge", driverSpace, 16, 4, 0);
        assertInstanceInSearchSpace("c5.2xlarge", driverSpace, 16, 8, 0);
        Assert.assertNull(driverSpace.get(GBtoBytes(32)));
        // assertions for executor space
        InstanceSearchSpace executorSpace = interestBasedEnumerator.getExecutorSpace();
        assertEquals(3, executorSpace.size());
        assertInstanceInSearchSpace("c5.xlarge", executorSpace, 8, 4, 0);
        assertInstanceInSearchSpace("m5.xlarge", executorSpace, 16, 4, 0);
        assertInstanceInSearchSpace("c5.2xlarge", executorSpace, 16, 8, 0);
        assertInstanceInSearchSpace("m5.2xlarge", executorSpace, 32, 8, 0);
    }

    @Test
    public void preprocessingInterestBasedBroadcastMemoryTest() {
        Enumerator interestBasedEnumerator = getInterestBasedEnumeratorPrebuild()
                .withFitDriverMemory(false)
                .withFitBroadcastMemory(true)
                .build();

        HashMap<String, CloudInstance> instances = TestingUtils.getSimpleCloudInstanceMap();
        interestBasedEnumerator.setInstanceTable(instances);

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
        assertInstanceInSearchSpace("c5.xlarge", executorSpace, 8, 4, 0);
        assertInstanceInSearchSpace("m5.xlarge", executorSpace, 16, 4, 0);
        assertInstanceInSearchSpace("c5.2xlarge", executorSpace, 16, 8, 0);
        Assert.assertNull(executorSpace.get(GBtoBytes(32)));
    }

    @Test
    public void evaluateSingleNodeExecutionGridBasedTest() {
        Enumerator gridBasedEnumerator;
        boolean result;

        gridBasedEnumerator = getGridBasedEnumeratorPrebuild()
                .withNumberExecutorsRange(0,1)
                .build();

        // memory not relevant for grid-based enumerator
        result = gridBasedEnumerator.evaluateSingleNodeExecution(-1);
        Assert.assertTrue(result);

        gridBasedEnumerator = getGridBasedEnumeratorPrebuild()
                .withNumberExecutorsRange(1,2)
                .build();

        // memory not relevant for grid-based enumerator
        result = gridBasedEnumerator.evaluateSingleNodeExecution(-1);
        Assert.assertFalse(result);
    }

    @Test
    public void estimateRangeExecutorsGridBasedStepSizeTest() {
        Enumerator gridBasedEnumerator;
        ArrayList<Integer> expectedResult;
        ArrayList<Integer> actualResult;

        // num. executors range starting from zero and step size = 2
        gridBasedEnumerator = getGridBasedEnumeratorPrebuild()
                .withNumberExecutorsRange(0, 10)
                .withStepSizeExecutor(2)
                .build();
        // test the general case when the max level of parallelism is not reached (0 is never part of the result)
        expectedResult = new ArrayList<>(List.of(2, 4, 6, 8, 10));
        actualResult = gridBasedEnumerator.estimateRangeExecutors(-1, 4);
        Assert.assertEquals(expectedResult, actualResult);
        // test the case when the max level of parallelism (1000) is reached (0 is never part of the result)
        expectedResult = new ArrayList<>(List.of(2, 4));
        actualResult = gridBasedEnumerator.estimateRangeExecutors(-1, 200);
        Assert.assertEquals(expectedResult, actualResult);

        // num. executors range not starting from zero and without step size given
        gridBasedEnumerator = getGridBasedEnumeratorPrebuild()
                .withNumberExecutorsRange(3, 8)
                .build();
        // test the general case when the max level of parallelism is not reached (0 is never part of the result)
        expectedResult = new ArrayList<>(List.of(3, 4, 5, 6, 7, 8));
        actualResult = gridBasedEnumerator.estimateRangeExecutors(-1, 4);
        Assert.assertEquals(expectedResult, actualResult);
        // test the case when the max level of parallelism (1000) is reached (0 is never part of the result)
        expectedResult = new ArrayList<>(List.of(3, 4, 5));
        actualResult = gridBasedEnumerator.estimateRangeExecutors(-1, 200);
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
        actualResult = gridBasedEnumerator.estimateRangeExecutors(-1, 4);
        Assert.assertEquals(expectedResult, actualResult);
        // test the case when the max level of parallelism (1000) is reached (0 is never part of the result)
        expectedResult = new ArrayList<>(List.of(1, 2, 4));
        actualResult = gridBasedEnumerator.estimateRangeExecutors(-1, 200);
        Assert.assertEquals(expectedResult, actualResult);

        // num. executors range not starting from zero and with exponential base = 3
        gridBasedEnumerator = getGridBasedEnumeratorPrebuild()
                .withNumberExecutorsRange(3, 30)
                .withExpBaseExecutors(3)
                .build();
        // test the general case when the max level of parallelism is not reached (0 is never part of the result)
        expectedResult = new ArrayList<>(List.of(3,9, 27));
        actualResult = gridBasedEnumerator.estimateRangeExecutors(-1, 4);
        Assert.assertEquals(expectedResult, actualResult);
        // test the case when the max level of parallelism (1000) is reached (0 is never part of the result)
        expectedResult = new ArrayList<>(List.of(3,9));
        actualResult = gridBasedEnumerator.estimateRangeExecutors(-1, 100);
        Assert.assertEquals(expectedResult, actualResult);
    }

    @Test
    public void evaluateSingleNodeExecutionInterestBasedTest() {
        boolean result;

        // no fitting the memory estimates for checkpointing
        Enumerator interestBasedEnumerator = getInterestBasedEnumeratorPrebuild()
                .withNumberExecutorsRange(0, 5)
                .withFitDriverMemory(false)
                .withFitBroadcastMemory(false)
                .withCheckSingleNodeExecution(true)
                .build();

        HashMap<String, CloudInstance> instances = TestingUtils.getSimpleCloudInstanceMap();
        interestBasedEnumerator.setInstanceTable(instances);

        TreeSet<Long> mockingMemoryEstimates = new TreeSet<>(Set.of(GBtoBytes(6), GBtoBytes(12)));
        try (MockedStatic<InterestBasedEnumerator> mockedEnumerator =
                     Mockito.mockStatic(InterestBasedEnumerator.class, Mockito.CALLS_REAL_METHODS)) {
            mockedEnumerator
                    .when(() -> InterestBasedEnumerator.getMemoryEstimates(
                            any(Program.class),
                            eq(false),
                            eq(OptimizerUtils.MEM_UTIL_FACTOR)))
                    .thenReturn(mockingMemoryEstimates);
            // initiate memoryEstimatesSpark
            interestBasedEnumerator.preprocessing();
        }

        result = interestBasedEnumerator.evaluateSingleNodeExecution(GBtoBytes(8));
        Assert.assertFalse(result);
    }

    @Test
    public void estimateRangeExecutorsInterestBasedGeneralTest() {
        ArrayList<Integer> expectedResult;
        ArrayList<Integer>actualResult;

        // no fitting the memory estimates for checkpointing
        Enumerator interestBasedEnumerator = getInterestBasedEnumeratorPrebuild()
                .withNumberExecutorsRange(0, 5)
                .build();
        // test the general case when the max level of parallelism is not reached (0 is never part of the result)
        expectedResult = new ArrayList<>(List.of(1, 2, 3, 4, 5));
        actualResult = interestBasedEnumerator.estimateRangeExecutors(-1, 4);
        Assert.assertEquals(expectedResult, actualResult);
        // test the case when the max level of parallelism (1000) is reached (0 is never part of the result)
        expectedResult = new ArrayList<>(List.of(1, 2, 3));
        actualResult = interestBasedEnumerator.estimateRangeExecutors(-1, 256);
        Assert.assertEquals(expectedResult, actualResult);
    }

    @Test
    public void estimateRangeExecutorsInterestBasedCheckpointMemoryTest() {
        ArrayList<Integer> expectedResult;
        ArrayList<Integer>actualResult;

        // fitting the memory estimates for checkpointing
        Enumerator interestBasedEnumerator = getInterestBasedEnumeratorPrebuild()
                .withNumberExecutorsRange(0, 5)
                .withFitCheckpointMemory(true)
                .withFitDriverMemory(false)
                .withFitBroadcastMemory(false)
                .build();

        HashMap<String, CloudInstance> instances = TestingUtils.getSimpleCloudInstanceMap();
        interestBasedEnumerator.setInstanceTable(instances);

        TreeSet<Long> mockingMemoryEstimates = new TreeSet<>(Set.of(GBtoBytes(20), GBtoBytes(40)));
        try (MockedStatic<InterestBasedEnumerator> mockedEnumerator =
                     Mockito.mockStatic(InterestBasedEnumerator.class, Mockito.CALLS_REAL_METHODS)) {
            mockedEnumerator
                    .when(() -> InterestBasedEnumerator.getMemoryEstimates(
                            any(Program.class),
                            eq(true),
                            eq(InterestBasedEnumerator.BROADCAST_MEMORY_FACTOR)))
                    .thenReturn(mockingMemoryEstimates);
            // initiate memoryEstimatesSpark
            interestBasedEnumerator.preprocessing();
        }

        // test the general case when the max level of parallelism is not reached (0 is never part of the result)
        expectedResult = new ArrayList<>(List.of(1, 2, 3));
        actualResult = interestBasedEnumerator.estimateRangeExecutors(GBtoBytes(16), 4);
        Assert.assertEquals(expectedResult, actualResult);
        // test the case when the max level of parallelism (1000) is reached (0 is never part of the result)
        expectedResult = new ArrayList<>(List.of(1, 2));
        actualResult = interestBasedEnumerator.estimateRangeExecutors(GBtoBytes(16), 500);
        Assert.assertEquals(expectedResult, actualResult);
    }

    @Test
    public void processingTest() {
        // all implemented enumerators should enumerate the same solution pool in this basic case - empty program
        Enumerator gridBasedEnumerator = getGridBasedEnumeratorPrebuild()
                .withTimeLimit(Double.MAX_VALUE)
                .withNumberExecutorsRange(0, 2)
                .build();

        Enumerator interestBasedEnumerator = getInterestBasedEnumeratorPrebuild()
                .withNumberExecutorsRange(0, 2)
                .build();

        HashMap<String, CloudInstance> instances = TestingUtils.getSimpleCloudInstanceMap();
        InstanceSearchSpace space = new InstanceSearchSpace();
        space.initSpace(instances);

        // run processing for the grid based enumerator
        gridBasedEnumerator.setDriverSpace(space);
        gridBasedEnumerator.setExecutorSpace(space);
        gridBasedEnumerator.processing();
        ArrayList<SolutionPoint> actualSolutionPoolGB = gridBasedEnumerator.getSolutionPool();
        // run processing for the interest based enumerator
        interestBasedEnumerator.setDriverSpace(space);
        interestBasedEnumerator.setExecutorSpace(space);
        interestBasedEnumerator.processing();
        ArrayList<SolutionPoint> actualSolutionPoolIB = gridBasedEnumerator.getSolutionPool();


        ArrayList<CloudInstance> expectedInstances = new ArrayList<>(Arrays.asList(
                instances.get("c5.xlarge"),
                instances.get("m5.xlarge")
        ));
        // expected solution pool with 0 executors (number executors = 0, executors and executorInstance being null)
        // each solution having one of the available instances as driver node
        Assert.assertEquals(expectedInstances.size(), actualSolutionPoolGB.size());
        Assert.assertEquals(expectedInstances.size(), actualSolutionPoolIB.size());
        for (int i = 0; i < expectedInstances.size(); i++) {
            SolutionPoint pointGB = actualSolutionPoolGB.get(i);
            Assert.assertEquals(0, pointGB.numberExecutors);
            Assert.assertEquals(expectedInstances.get(i), pointGB.driverInstance);
            Assert.assertNull(pointGB.executorInstance);
            SolutionPoint pointIB = actualSolutionPoolGB.get(i);
            Assert.assertEquals(0, pointIB.numberExecutors);
            Assert.assertEquals(expectedInstances.get(i), pointIB.driverInstance);
            Assert.assertNull(pointIB.executorInstance);
        }
    }

    @Test
    public void postprocessingTest() {
        // postprocessing equivalent for all types of enumerators
        Enumerator enumerator = getGridBasedEnumeratorPrebuild().build();
        // construct solution pool
        // first dummy configuration point since not relevant for postprocessing
        ConfigurationPoint dummyPoint = new ConfigurationPoint(null);
        SolutionPoint solution1 = new SolutionPoint(dummyPoint, 1000, 1000);
        SolutionPoint solution2 = new SolutionPoint(dummyPoint, 900, 1000); // optimal point
        SolutionPoint solution3 = new SolutionPoint(dummyPoint, 800, 10000);
        SolutionPoint solution4 = new SolutionPoint(dummyPoint, 1000, 10000);
        SolutionPoint solution5 = new SolutionPoint(dummyPoint, 900, 10000);
        ArrayList<SolutionPoint> mockListSolutions = new ArrayList<>(List.of(solution1, solution2, solution3, solution4, solution5));
        enumerator.setSolutionPool(mockListSolutions);

        SolutionPoint optimalSolution = enumerator.postprocessing();
        assertEquals(solution2, optimalSolution);
    }

    @Test
    public void GridBasedEnumerationMinPriceTest() {
        Enumerator gridBasedEnumerator = getGridBasedEnumeratorPrebuild()
                .withNumberExecutorsRange(0, 2)
                .build();

        gridBasedEnumerator.setInstanceTable(TestingUtils.getSimpleCloudInstanceMap());

        gridBasedEnumerator.preprocessing();
        gridBasedEnumerator.processing();
        SolutionPoint solution = gridBasedEnumerator.postprocessing();

        // expected m5.xlarge since it is the cheaper
        Assert.assertEquals("m5.xlarge", solution.driverInstance.getInstanceName());
        // expected no executor nodes since tested for a 'zero' program
        Assert.assertEquals(0, solution.numberExecutors);
    }

    @Test
    public void InterestBasedEnumerationMinPriceTest() {
        Enumerator interestBasedEnumerator = getInterestBasedEnumeratorPrebuild()
                .withNumberExecutorsRange(0, 2)
                .build();

        interestBasedEnumerator.setInstanceTable(TestingUtils.getSimpleCloudInstanceMap());

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
                .withOptimizationStrategy(Enumerator.OptimizationStrategy.MinTime)
                .withBudget(Double.MAX_VALUE)
                .withNumberExecutorsRange(0, 2)
                .build();

        gridBasedEnumerator.setInstanceTable(TestingUtils.getSimpleCloudInstanceMap());

        gridBasedEnumerator.preprocessing();
        gridBasedEnumerator.processing();
        SolutionPoint solution = gridBasedEnumerator.postprocessing();

        // expected m5.xlarge since it is the cheaper
        Assert.assertEquals("m5.xlarge", solution.driverInstance.getInstanceName());
        // expected no executor nodes since tested for a 'zero' program
        Assert.assertEquals(0, solution.numberExecutors);
    }

    @Test
    public void InterestBasedEnumerationMinTimeTest() {
        Enumerator interestBasedEnumerator = getInterestBasedEnumeratorPrebuild()
                .withOptimizationStrategy(Enumerator.OptimizationStrategy.MinTime)
                .withBudget(Double.MAX_VALUE)
                .withNumberExecutorsRange(0, 2)
                .build();

        interestBasedEnumerator.setInstanceTable(TestingUtils.getSimpleCloudInstanceMap());

        interestBasedEnumerator.preprocessing();
        interestBasedEnumerator.processing();
        SolutionPoint solution = interestBasedEnumerator.postprocessing();

        // expected c5.xlarge since is the instance with at least memory
        Assert.assertEquals("c5.xlarge", solution.driverInstance.getInstanceName());
        // expected no executor nodes since tested for a 'zero' program
        Assert.assertEquals(0, solution.numberExecutors);
    }

    // Helpers
    private static Enumerator.Builder getGridBasedEnumeratorPrebuild() {
        Program emptyProgram = new Program();
        return (new Enumerator.Builder())
                .withRuntimeProgram(emptyProgram)
                .withEnumerationStrategy(Enumerator.EnumerationStrategy.GridBased)
                .withOptimizationStrategy(Enumerator.OptimizationStrategy.MinPrice)
                .withTimeLimit(Double.MAX_VALUE);
    }

    private static Enumerator.Builder getInterestBasedEnumeratorPrebuild() {
        Program emptyProgram = new Program();
        return (new Enumerator.Builder())
                .withRuntimeProgram(emptyProgram)
                .withEnumerationStrategy(Enumerator.EnumerationStrategy.InterestBased)
                .withOptimizationStrategy(Enumerator.OptimizationStrategy.MinPrice)
                .withTimeLimit(Double.MAX_VALUE);
    }

    private static void assertInstanceInSearchSpace(
            String expectedName,
            InstanceSearchSpace searchSpace,
            int memory, /* in GB */
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
