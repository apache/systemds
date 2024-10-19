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
import java.util.Map;

import static org.apache.sysds.resource.ResourceOptimizer.createOptions;
import static org.apache.sysds.resource.ResourceOptimizer.initEnumerator;
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
        Map<String, String> envOptions = generateTestingEnvOptionsRequired("any");

        Enumerator actualEnumerator = assertProperEnumeratorInitialization(args, envOptions);
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
        File dmlScript = generateTmpDMLScript("m = $1;", "n = $2;");

        String[] args = {
                "-f", dmlScript.getPath(),
                "-args", "10", "100"
        };
        Map<String, String> envOptions = generateTestingEnvOptionsRequired("any");

        assertProperEnumeratorInitialization(args, envOptions);

        Files.deleteIfExists(dmlScript.toPath());
    }

    @Test
    public void initEnumeratorFromArgsWithNvargTest() throws IOException {
        File dmlScript = generateTmpDMLScript("m = $m;", "n = $n;");

        String[] args = {
                "-f", dmlScript.getPath(),
                "-nvargs", "m=10", "n=100"
        };
        Map<String, String> envOptions = generateTestingEnvOptionsRequired("any");

        assertProperEnumeratorInitialization(args, envOptions);

        Files.deleteIfExists(dmlScript.toPath());
    }

    @Test
    public void initEnumeratorMinTimeOptimizationInvalidTest() {
        String[] args = {
                "-f", HOME+"Algorithm_L2SVM.dml",
        };
        Options options = createOptions();
        CommandLineParser clParser = new PosixParser();
        CommandLine line = null;
        try {
            line = clParser.parse(options, args);
        } catch (ParseException e) {
            Assert.fail("ParseException should not have been raise here: "+e);
        }
        Map<String, String> invalidEnvOption = generateTestingEnvOptionsRequired("any");
        invalidEnvOption.put("OPTIMIZATION_FUNCTION", "time");
        try {
            initEnumerator(line, options, invalidEnvOption);
            Assert.fail("ParseException should not have been raise here for not provided -maxPrice argument");
        } catch (Exception e) {
            Assert.assertTrue(e instanceof ParseException);
        }


        String[] validArgs = {
                "-f", HOME+"Algorithm_L2SVM.dml",
        };
        Map<String, String> validEnvOptions = generateTestingEnvOptionsRequired("any");
        validEnvOptions.put("OPTIMIZATION_FUNCTION", "time");
        validEnvOptions.put("MAX_PRICE", "1000");
        Enumerator actualEnumerator = assertProperEnumeratorInitialization(validArgs, validEnvOptions);
        Assert.assertEquals(Enumerator.OptimizationStrategy.MinTime, actualEnumerator.getOptStrategy());
        Assert.assertEquals(1000, actualEnumerator.getMaxPrice(), 0.0);
    }

    @Test
    public void initEnumeratorMinPriceOptimizationInvalidTest() {
        String[] args = {
                "-f", HOME+"Algorithm_L2SVM.dml",
        };
        Options options = createOptions();
        CommandLineParser clParser = new PosixParser();
        CommandLine line = null;
        try {
            line = clParser.parse(options, args);
        } catch (ParseException e) {
            Assert.fail("ParseException should not have been raise here: "+e);
        }
        Map<String, String> invalidEnvOptions = generateTestingEnvOptionsRequired("any");
        invalidEnvOptions.put("OPTIMIZATION_FUNCTION", "price");
        try {
            initEnumerator(line, options, invalidEnvOptions);
            Assert.fail("ParseException should not have been raise here for not provided -maxTime argument");
        } catch (Exception e) {
            Assert.assertTrue(e instanceof ParseException);
        }


        String[] validArgs = {
                "-f", HOME+"Algorithm_L2SVM.dml",
        };
        Map<String, String> validEnvOptions = generateTestingEnvOptionsRequired("any");
        validEnvOptions.put("OPTIMIZATION_FUNCTION", "price");
        validEnvOptions.put("MAX_TIME", "1000");
        Enumerator actualEnumerator = assertProperEnumeratorInitialization(validArgs, validEnvOptions);
        Assert.assertEquals(Enumerator.OptimizationStrategy.MinPrice, actualEnumerator.getOptStrategy());
        Assert.assertEquals(1000, actualEnumerator.getMaxTime(), 0.0);
    }

    @Test
    public void initGridEnumeratorWithAllOptionalArgsTest() {
        String[] args = {
                "-f", HOME+"Algorithm_L2SVM.dml",
        };
        Map<String, String> envOptions = generateTestingEnvOptionsRequired("any");
        envOptions.put("ENUMERATION", "grid");
        envOptions.put("STEP_SIZE", "3");
        envOptions.put("EXPONENTIAL_BASE", "2");

        Enumerator actualEnumerator = assertProperEnumeratorInitialization(args, envOptions);
        Assert.assertTrue(actualEnumerator instanceof GridBasedEnumerator);
        // assert enum. specific default
        Assert.assertEquals(3, ((GridBasedEnumerator) actualEnumerator).getStepSize());
        Assert.assertEquals(2, ((GridBasedEnumerator) actualEnumerator).getExpBase());
    }

    @Test
    public void initInterestEnumeratorWithDefaultsTest() {
        String[] args = {
                "-f", HOME+"Algorithm_L2SVM.dml",
        };
        Map<String, String> envOptions = generateTestingEnvOptionsRequired("any");
        envOptions.put("ENUMERATION", "interest");

        Enumerator actualEnumerator = assertProperEnumeratorInitialization(args, envOptions);
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
        };
        Map<String, String> envOptions = generateTestingEnvOptionsRequired("any");
        envOptions.put("ENUMERATION", "prune");

        Enumerator actualEnumerator = assertProperEnumeratorInitialization(args, envOptions);
        Assert.assertTrue(actualEnumerator instanceof PruneBasedEnumerator);
    }

    @Test
    public void initInterestEnumeratorWithWithAllOptionsTest() {
        // set all the flags to opposite values to their defaults
        String[] args = {
                "-f", HOME+"Algorithm_L2SVM.dml",
        };
        Map<String, String> envOptions = generateTestingEnvOptionsRequired("any");
        envOptions.put("ENUMERATION", "interest");
        envOptions.put("USE_LARGEST_ESTIMATE", "false");
        envOptions.put("USE_CP_ESTIMATES", "false");
        envOptions.put("USE_BROADCASTS", "false");
        envOptions.put("USE_OUTPUTS", "true");

        InterestBasedEnumerator actualEnumerator =
                (InterestBasedEnumerator) assertProperEnumeratorInitialization(args, envOptions);
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
        };
        Map<String, String> envOptions = generateTestingEnvOptionsRequired("any");
        envOptions.put("INSTANCE_FAMILIES", "m5");
        envOptions.put("INSTANCE_SIZES", "2xlarge");

        Enumerator actualEnumerator = assertProperEnumeratorInitialization(args, envOptions);

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
        };
        Map<String, String> envOptions = generateTestingEnvOptionsRequired("any");
        envOptions.put("CPU_QUOTA", "256");

        Enumerator actualEnumerator = assertProperEnumeratorInitialization(args, envOptions);

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
    public void executeForL2SVM_MinimalSearchSpace_Test() throws IOException, ParseException {
        Path tmpOutFolder = Files.createTempDirectory("out");

        String[] args = {
                "-f", HOME+"Algorithm_L2SVM.dml",
                "-nvargs", "m=200000", "n=10000"
        };
        Options options = createOptions();
        CommandLineParser clParser = new PosixParser();
        CommandLine line = null;
        try {
            line = clParser.parse(options, args);
        } catch (ParseException e) {
            Assert.fail("ParseException should not have been raise here: "+e);
        }
        Map<String, String> envOptions = generateTestingEnvOptionsRequired(tmpOutFolder.toString());
        envOptions.put("MAX_EXECUTORS", "10");

        ResourceOptimizer.execute(line, options, envOptions);

        if (!DEBUG) {
            deleteDirectoryWithFiles(tmpOutFolder);
        }
    }

    @Test
    public void executeForL2SVM_MinimalSearchSpace_C5_XLARGE_Test() throws IOException, ParseException {
        Path tmpOutFolder = Files.createTempDirectory("out");

        String[] args = {
                "-f", HOME+"Algorithm_L2SVM.dml",
                "-nvargs", "m=200000", "n=10000"
        };
        Options options = createOptions();
        CommandLineParser clParser = new PosixParser();
        CommandLine line = null;
        try {
            line = clParser.parse(options, args);
        } catch (ParseException e) {
            Assert.fail("ParseException should not have been raise here: "+e);
        }
        Map<String, String> envOptions = generateTestingEnvOptionsRequired(tmpOutFolder.toString());
        envOptions.put("MAX_EXECUTORS", "10");
        envOptions.put("INSTANCE_FAMILIES", "c5,c5d,c5n");
        envOptions.put("INSTANCE_SIZES", "xlarge");

        ResourceOptimizer.execute(line, options, envOptions);

        if (!DEBUG) {
            deleteDirectoryWithFiles(tmpOutFolder);
        }
    }

    @Test
    public void executeForL2SVM_FullSearchSpace_Test() throws IOException, ParseException {
        Path tmpOutFolder = Files.createTempDirectory("out");

        String[] args = {
                "-f", HOME+"Algorithm_L2SVM.dml",
                "-nvargs", "m=10000", "n=10000"
        };
        Options options = createOptions();
        CommandLineParser clParser = new PosixParser();
        CommandLine line = null;
        try {
            line = clParser.parse(options, args);
        } catch (ParseException e) {
            Assert.fail("ParseException should not have been raise here: "+e);
        }
        Map<String, String> envOptions = generateTestingEnvOptionsRequired(tmpOutFolder.toString());
        envOptions.put("INFO_TABLE", DEFAULT_INSTANCE_INFO_TABLE);
        envOptions.put("MAX_EXECUTORS", "10");

        ResourceOptimizer.execute(line, options, envOptions);

        if (!DEBUG) {
            deleteDirectoryWithFiles(tmpOutFolder);
        }
    }

    @Test
    public void executeForReadAndWrite_Test() throws IOException, ParseException {
        Path tmpOutFolder = Files.createTempDirectory("out");

        String[] args = {
                "-f", HOME+"ReadAndWrite.dml",
                "-nvargs",
                    "fileA=s3://data/in/A.csv",
                    "fileA_Csv=s3://data/out/A.csv",
                    "fileA_Text=s3://data/out/A.txt"
        };
        Options options = createOptions();
        CommandLineParser clParser = new PosixParser();
        CommandLine line = null;
        try {
            line = clParser.parse(options, args);
        } catch (ParseException e) {
            Assert.fail("ParseException should not have been raise here: "+e);
        }
        Map<String, String> envOptions = generateTestingEnvOptionsRequired(tmpOutFolder.toString());
        envOptions.put("MAX_EXECUTORS", "2");
        String localInputs = "s3://data/in/A.csv=" + HOME + "data/A.csv";
        envOptions.put("LOCAL_INPUTS", localInputs);

        ResourceOptimizer.execute(line, options, envOptions);

        if (!DEBUG) {
            deleteDirectoryWithFiles(tmpOutFolder);
        }
    }

    // Helpers ---------------------------------------------------------------------------------------------------------

    private Enumerator assertProperEnumeratorInitialization(String[] args, Map<String, String> envOptions) {
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
            actualEnumerator = initEnumerator(line, options, envOptions);
        } catch (Exception e) {
            Assert.fail("Any exception should not have been raise here: "+e);
        }
        Assert.assertNotNull(actualEnumerator);

        return actualEnumerator;
    }
}
