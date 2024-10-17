package org.apache.sysds.test.component.resource;

import org.apache.commons.cli.*;
import org.apache.sysds.resource.CloudInstance;
import org.apache.sysds.resource.ResourceOptimizer;
import org.apache.sysds.resource.enumeration.Enumerator;
import org.apache.sysds.resource.enumeration.GridBasedEnumerator;
import org.apache.sysds.resource.enumeration.InterestBasedEnumerator;
import org.apache.sysds.resource.enumeration.PruneBasedEnumerator;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Assert;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;

import static org.apache.sysds.resource.ResourceOptimizer.createOptions;
import static org.apache.sysds.resource.ResourceOptimizer.initEnumeratorFromArgs;
import static org.apache.sysds.test.component.resource.ResourceTestUtils.*;

public class ResourceOptimizerTest extends AutomatedTestBase {
    private static final String TEST_DIR = "component/resource/";
    private static final String HOME = SCRIPT_DIR + TEST_DIR;

    @Override
    public void setUp() {}

    @Test
    public void initEnumeratorFromArgsDefaultsTest() {
        String[] args = {
                "-f", HOME+"Algorithm_L2SVM.dml",
        };
        Enumerator actualEnumerator = assertProperEnumeratorInitialization(args);
        Assert.assertTrue(actualEnumerator instanceof GridBasedEnumerator);
        // assert all defaults
        HashMap<String, CloudInstance> expectedInstances = getSimpleCloudInstanceMap();
        HashMap<String, CloudInstance> actualInstances = actualEnumerator.getInstances();
        for (String instanceName: expectedInstances.keySet()) {
            assertEqualsCloudInstances(expectedInstances.get(instanceName), actualInstances.get(instanceName));
        }
        Assert.assertEquals(Enumerator.EnumerationStrategy.GridBased, actualEnumerator.getEnumStrategy());
        Assert.assertEquals(Enumerator.OptimizationStrategy.MinCosts, actualEnumerator.getOptStrategy());
        // assert enum. specific default
        GridBasedEnumerator gridBasedEnumerator = (GridBasedEnumerator) actualEnumerator;
        Assert.assertEquals(1, gridBasedEnumerator.getStepSize());
        Assert.assertEquals(-1, gridBasedEnumerator.getExpBase());
    }

    @Test
    public void initEnumeratorFromArgsWithArgNTest() throws IOException {
        File dmlScript = generateTmlDMLScript("m = $1;", "n = $2;");

        String[] args = {
                "-f", dmlScript.getPath(),
                "-args", "10", "100"
        };
        assertProperEnumeratorInitialization(args);

        Files.deleteIfExists(dmlScript.toPath());
    }

    @Test
    public void initEnumeratorFromArgsWithNvargTest() throws IOException {
        File dmlScript = generateTmlDMLScript("m = $m;", "n = $n;");

        String[] args = {
                "-f", dmlScript.getPath(),
                "-nvargs", "m=10", "n=100"
        };
        assertProperEnumeratorInitialization(args);

        Files.deleteIfExists(dmlScript.toPath());
    }

    @Test
    public void intEnumeratorWithMinTimeOptimizationTest() {
        String[] invalidArgs = {
                "-f", HOME+"Algorithm_L2SVM.dml",
                "-optimizeFor", "time",
        };
        Options options = createOptions();
        CommandLineParser clParser = new PosixParser();
        CommandLine line = null;
        try {
            line = clParser.parse(options, invalidArgs);
        } catch (ParseException e) {
            Assert.fail("ParseException should not have been raise here: "+e);
        }
        try {
            initEnumeratorFromArgs(line, options);
            Assert.fail("ParseException should not have been raise here for not provided -maxPrice argument");
        } catch (Exception e) {
            Assert.assertTrue(e instanceof ParseException);
        }


        String[] validArgs = {
                "-f", HOME+"Algorithm_L2SVM.dml",
                "-optimizeFor", "time",
                "-maxPrice", "1000"
        };
        Enumerator actualEnumerator = assertProperEnumeratorInitialization(validArgs);
        Assert.assertEquals(Enumerator.OptimizationStrategy.MinTime, actualEnumerator.getOptStrategy());
        Assert.assertEquals(1000, actualEnumerator.getMaxPrice(), 0.0);
    }

    @Test
    public void intEnumeratorWithMinPriceOptimizationTest() {
        String[] invalidArgs = {
                "-f", HOME+"Algorithm_L2SVM.dml",
                "-optimizeFor", "price",
        };
        Options options = createOptions();
        CommandLineParser clParser = new PosixParser();
        CommandLine line = null;
        try {
            line = clParser.parse(options, invalidArgs);
        } catch (ParseException e) {
            Assert.fail("ParseException should not have been raise here: "+e);
        }
        try {
            initEnumeratorFromArgs(line, options);
            Assert.fail("ParseException should not have been raise here for not provided -maxTime argument");
        } catch (Exception e) {
            Assert.assertTrue(e instanceof ParseException);
        }


        String[] validArgs = {
                "-f", HOME+"Algorithm_L2SVM.dml",
                "-optimizeFor", "price",
                "-maxTime", "1000"
        };
        Enumerator actualEnumerator = assertProperEnumeratorInitialization(validArgs);
        Assert.assertEquals(Enumerator.OptimizationStrategy.MinPrice, actualEnumerator.getOptStrategy());
        Assert.assertEquals(1000, actualEnumerator.getMaxTime(), 0.0);
    }

    @Test
    public void initGridEnumeratorWithAllOptionalArgsTest() {
        String[] args = {
                "-f", HOME+"Algorithm_L2SVM.dml",
                "-enum", "grid",
                "-stepSize", "3",
                "-expBase", "2"
        };
        Enumerator actualEnumerator = assertProperEnumeratorInitialization(args);
        Assert.assertTrue(actualEnumerator instanceof GridBasedEnumerator);
        // assert enum. specific default
        Assert.assertEquals(3, ((GridBasedEnumerator) actualEnumerator).getStepSize());
        Assert.assertEquals(2, ((GridBasedEnumerator) actualEnumerator).getExpBase());
    }

    @Test
    public void initInterestEnumeratorWithDefaultsTest() {
        String[] args = {
                "-f", HOME+"Algorithm_L2SVM.dml",
                "-enum", "interest",
        };
        Enumerator actualEnumerator = assertProperEnumeratorInitialization(args);
        Assert.assertTrue(actualEnumerator instanceof InterestBasedEnumerator);
        // assert enum. specific default
        Assert.assertTrue(((InterestBasedEnumerator) actualEnumerator).interestLargestEstimateEnabled());
        Assert.assertTrue(((InterestBasedEnumerator) actualEnumerator).interestEstimatesInCPEnabled());
        Assert.assertTrue(((InterestBasedEnumerator) actualEnumerator).interestBroadcastVars());
        Assert.assertFalse(((InterestBasedEnumerator) actualEnumerator).interestOutputCachingEnabled());

    }

    @Test
    public void initPruneEnumeratorWithDefaultsTest() {
        String[] args = {
                "-f", HOME+"Algorithm_L2SVM.dml",
                "-enum", "prune",
        };
        Enumerator actualEnumerator = assertProperEnumeratorInitialization(args);
        Assert.assertTrue(actualEnumerator instanceof PruneBasedEnumerator);
    }

    @Test
    public void initInterestEnumeratorWithWithAllOptionsTest() {
        // set all the flags to opposite values to their defaults
        String[] args = {
                "-f", HOME+"Algorithm_L2SVM.dml",
                "-enum", "interest",
                "-useLargestEst", "false",
                "-useCpEstimates", "false",
                "-useBroadcasts", "false",
                "-useOutputs", "true"
        };
        InterestBasedEnumerator actualEnumerator = (InterestBasedEnumerator) assertProperEnumeratorInitialization(args);
        // assert enum. specific default
        Assert.assertFalse(actualEnumerator.interestLargestEstimateEnabled());
        Assert.assertFalse(actualEnumerator.interestEstimatesInCPEnabled());
        Assert.assertFalse(actualEnumerator.interestBroadcastVars());
        Assert.assertTrue(actualEnumerator.interestOutputCachingEnabled());
    }

    @Test
    public void initEnumeratorWithInstanceRangeTest() {
        String[] args = {
                "-f", HOME+"Algorithm_L2SVM.dml",
                "-instanceFamilies", "m5",
                "-instanceSizes", "2xlarge"
        };
        Enumerator actualEnumerator = assertProperEnumeratorInitialization(args);

        HashMap<String, CloudInstance> inputInstances = getSimpleCloudInstanceMap();
        HashMap<String, CloudInstance> expectedInstances = new HashMap<>();
        expectedInstances.put("m5.2xlarge", inputInstances.get("m5.2xlarge"));

        HashMap<String, CloudInstance> actualInstances = actualEnumerator.getInstances();

        for (String instanceName: expectedInstances.keySet()) {
            assertEqualsCloudInstances(expectedInstances.get(instanceName), actualInstances.get(instanceName));
        }
    }

    @Test
    public void initEnumeratorWithCustomCPUQuotaTest() {
        String[] args = {
                "-f", HOME+"Algorithm_L2SVM.dml",
                "-quotaCPU", "256",
        };
        Enumerator actualEnumerator = assertProperEnumeratorInitialization(args);

        ArrayList<Integer> actualRange = actualEnumerator.estimateRangeExecutors(128, -1, 16);
        Assert.assertEquals(actualRange.size(), 8);
        Assert.assertEquals(8, (int) actualRange.get(7));
    }

    @Test
    public void mainWithHelpArgTest() {
        // test with valid argument combination
        String[] validArgs = {
                "-help"
        };
        try {
            ResourceOptimizer.main(validArgs);
        } catch (Exception e) {
            Assert.fail("Passing only '-help' should never raise an exception, but the following one was raised: "+e);
        }

        // test with invalid argument combination
        String[] invalidArgs = {
                "-help",
                "-f", HOME+"Algorithm_L2SVM.dml",
        };
        try {
            ResourceOptimizer.main(invalidArgs);
            Assert.fail("Passing '-help' and '-f' is not a valid combination but no exception was raised");
        } catch (Exception e) {
            Assert.assertTrue(e instanceof ParseException);
        }
    }

    @Test
    public void mainForL2SVM_MinimalSearchSpace_Test() throws IOException, ParseException {
        File tmpRegionFile = ResourceTestUtils.generateMinimalFeeTableFile();
        File tmpInfoFile = ResourceTestUtils.generateMinimalInstanceInfoTableFile();
        Path tmpOutFolder = Files.createTempDirectory("out");

        // TODO: fix why for "-nvargs", "m=10000000", "n=100000" time and price is lower than for "-nvargs", "m=1000000", "n=100000"
        String[] args = {
                "-f", HOME+"Algorithm_L2SVM.dml",
                "-infoTable", tmpInfoFile.getPath(),
                "-region", TEST_REGION,
                "-regionTable", tmpRegionFile.getPath(),
                "-output", tmpOutFolder.toString(),
                "-maxExecutors", "10",
                "-nvargs", "m=200000", "n=10000"
        };
        ResourceOptimizer.main(args);

        Files.deleteIfExists(tmpRegionFile.toPath());
        Files.deleteIfExists(tmpInfoFile.toPath());
        // TODO: delete all files and the folder 'tmpOutFolder'
    }

    @Test
    public void mainForL2SVM_MinimalSearchSpace_C5_XLARGE_Test() throws IOException, ParseException {
        File tmpRegionFile = ResourceTestUtils.generateMinimalFeeTableFile();
        File tmpInfoFile = ResourceTestUtils.generateMinimalInstanceInfoTableFile();
        Path tmpOutFolder = Files.createTempDirectory("out");

        // TODO: fix why for "-nvargs", "m=10000000", "n=100000" time and price is lower than for "-nvargs", "m=1000000", "n=100000"
        String[] args = {
                "-f", HOME+"Algorithm_L2SVM.dml",
                "-infoTable", tmpInfoFile.getPath(),
                "-region", TEST_REGION,
                "-regionTable", tmpRegionFile.getPath(),
                "-output", tmpOutFolder.toString(),
                "-maxExecutors", "10",
                "-instanceFamilies", "c5", "c5d", "c5n",
                "-instanceSizes", "xlarge",
                "-nvargs", "m=200000", "n=10000"
        };
        ResourceOptimizer.main(args);

        Files.deleteIfExists(tmpRegionFile.toPath());
        Files.deleteIfExists(tmpInfoFile.toPath());
        // TODO: delete all files and the folder 'tmpOutFolder'
    }

    @Test
    public void mainForL2SVM_FullSearchSpace_Test() throws IOException, ParseException {
        File tmpRegionFile = ResourceTestUtils.generateMinimalFeeTableFile();
        File tmpInfoFile = ResourceTestUtils.generateMinimalInstanceInfoTableFile();
        Path tmpOutFolder = Files.createTempDirectory("out");

        // TODO: fix why for "-nvargs", "m=10000000", "n=100000" time and price is lower than for "-nvargs", "m=1000000", "n=100000"
        String[] args = {
                "-f", HOME+"Algorithm_L2SVM.dml",
                "-infoTable", tmpInfoFile.getPath(),
                "-region", TEST_REGION,
                "-regionTable", tmpRegionFile.getPath(),
                "-output", tmpOutFolder.toString(),
                "-maxExecutors", "10",
                "-nvargs", "m=10000", "n=10000"
        };
        ResourceOptimizer.main(args);

        Files.deleteIfExists(tmpRegionFile.toPath());
        Files.deleteIfExists(tmpInfoFile.toPath());
        // TODO: delete all files and the folder 'tmpOutFolder'
    }

    // Helpers ---------------------------------------------------------------------------------------------------------

    private Enumerator assertProperEnumeratorInitialization(String[] args) {
        Options options = createOptions();
        CommandLineParser clParser = new PosixParser();
        CommandLine line = null;
        try {
            line = clParser.parse(options, args);
        } catch (ParseException e) {
            Assert.fail("ParseException should not have been raise here: "+e);
        }
        Enumerator actualEnumerator = null;
        try {
            actualEnumerator = initEnumeratorFromArgs(line, options);
        } catch (Exception e) {
            Assert.fail("Any exception should not have been raise here: "+e);
        }
        Assert.assertNotNull(actualEnumerator);

        return actualEnumerator;
    }
}
