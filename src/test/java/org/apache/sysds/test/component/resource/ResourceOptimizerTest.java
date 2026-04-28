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

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.configuration2.PropertiesConfiguration;
import org.apache.sysds.resource.CloudInstance;
import org.apache.sysds.resource.ResourceOptimizer;
import org.apache.sysds.resource.enumeration.Enumerator;
import org.apache.sysds.resource.enumeration.GridBasedEnumerator;
import org.apache.sysds.resource.enumeration.InterestBasedEnumerator;
import org.apache.sysds.resource.enumeration.PruneBasedEnumerator;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;

import static org.apache.sysds.resource.ResourceOptimizer.createOptions;
import static org.apache.sysds.resource.ResourceOptimizer.initEnumerator;

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
		PropertiesConfiguration options = ResourceTestUtils.generateTestingOptionsRequired("any");

		Enumerator actualEnumerator = assertProperEnumeratorInitialization(args, options);
		Assert.assertTrue(actualEnumerator instanceof GridBasedEnumerator);
		// assert all defaults
		HashMap<String, CloudInstance> expectedInstances = ResourceTestUtils.getSimpleCloudInstanceMap();
		HashMap<String, CloudInstance> actualInstances = actualEnumerator.getInstances();
		for (String instanceName: expectedInstances.keySet()) {
			ResourceTestUtils.assertEqualsCloudInstances(expectedInstances.get(instanceName), actualInstances.get(instanceName));
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
		File dmlScript = ResourceTestUtils.generateTmpDMLScript("m = $1;", "n = $2;");

		String[] args = {
				"-f", dmlScript.getPath(),
				"-args", "10", "100"
		};
		PropertiesConfiguration options = ResourceTestUtils.generateTestingOptionsRequired("any");

		assertProperEnumeratorInitialization(args, options);

		Files.deleteIfExists(dmlScript.toPath());
	}

	@Test
	public void initEnumeratorFromArgsWithNvargTest() throws IOException {
		File dmlScript = ResourceTestUtils.generateTmpDMLScript("m = $m;", "n = $n;");

		String[] args = {
				"-f", dmlScript.getPath(),
				"-nvargs", "m=10", "n=100"
		};
		PropertiesConfiguration options = ResourceTestUtils.generateTestingOptionsRequired("any");

		assertProperEnumeratorInitialization(args, options);

		Files.deleteIfExists(dmlScript.toPath());
	}

	@Test
	public void initEnumeratorCostsWeightOptimizationInvalidTest() {
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
		PropertiesConfiguration invalidOptions = ResourceTestUtils.generateTestingOptionsRequired("any");
		invalidOptions.setProperty("OPTIMIZATION_FUNCTION", "costs");
		invalidOptions.setProperty("COSTS_WEIGHT", "10");
		try {
			initEnumerator(line, invalidOptions);
			Assert.fail("ParseException should have been raise here for not provided MAX_PRICE option");
		} catch (Exception e) {
			Assert.assertTrue(e instanceof ParseException);
		}


		String[] validArgs = {
				"-f", HOME+"Algorithm_L2SVM.dml",
		};
		PropertiesConfiguration validOptions = ResourceTestUtils.generateTestingOptionsRequired("any");
		validOptions.setProperty("OPTIMIZATION_FUNCTION", "costs");
		validOptions.setProperty("COSTS_WEIGHT", "0.1");
		Enumerator actualEnumerator = assertProperEnumeratorInitialization(validArgs, validOptions);
		Assert.assertEquals(Enumerator.OptimizationStrategy.MinCosts, actualEnumerator.getOptStrategy());
		Assert.assertEquals(0.1, actualEnumerator.getCostsWeightFactor(), 0.0);
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
		PropertiesConfiguration invalidOptions = ResourceTestUtils.generateTestingOptionsRequired("any");
		invalidOptions.setProperty("OPTIMIZATION_FUNCTION", "time");
		try {
			initEnumerator(line, invalidOptions);
			Assert.fail("ParseException should have been raise here for not provided MAX_PRICE option");
		} catch (Exception e) {
			Assert.assertTrue(e instanceof ParseException);
		}


		String[] validArgs = {
				"-f", HOME+"Algorithm_L2SVM.dml",
		};
		PropertiesConfiguration validOptions = ResourceTestUtils.generateTestingOptionsRequired("any");
		validOptions.setProperty("OPTIMIZATION_FUNCTION", "time");
		validOptions.setProperty("MAX_PRICE", "1000");
		Enumerator actualEnumerator = assertProperEnumeratorInitialization(validArgs, validOptions);
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
		PropertiesConfiguration invalidOptions = ResourceTestUtils.generateTestingOptionsRequired("any");
		invalidOptions.setProperty("OPTIMIZATION_FUNCTION", "price");
		try {
			initEnumerator(line, invalidOptions);
			Assert.fail("ParseException should have been raise here for not provided MAX_TIME option");
		} catch (Exception e) {
			Assert.assertTrue(e instanceof ParseException);
		}


		String[] validArgs = {
				"-f", HOME+"Algorithm_L2SVM.dml",
		};
		PropertiesConfiguration validOptions = ResourceTestUtils.generateTestingOptionsRequired("any");
		validOptions.setProperty("OPTIMIZATION_FUNCTION", "price");
		validOptions.setProperty("MAX_TIME", "1000");
		Enumerator actualEnumerator = assertProperEnumeratorInitialization(validArgs, validOptions);
		Assert.assertEquals(Enumerator.OptimizationStrategy.MinPrice, actualEnumerator.getOptStrategy());
		Assert.assertEquals(1000, actualEnumerator.getMaxTime(), 0.0);
	}

	@Test
	public void initGridEnumeratorWithAllOptionalArgsTest() {
		String[] args = {
				"-f", HOME+"Algorithm_L2SVM.dml",
		};
		PropertiesConfiguration options = ResourceTestUtils.generateTestingOptionsRequired("any");
		options.setProperty("ENUMERATION", "grid");
		options.setProperty("STEP_SIZE", "3");
		options.setProperty("EXPONENTIAL_BASE", "2");

		Enumerator actualEnumerator = assertProperEnumeratorInitialization(args, options);
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
		PropertiesConfiguration options = ResourceTestUtils.generateTestingOptionsRequired("any");
		options.setProperty("ENUMERATION", "interest");

		Enumerator actualEnumerator = assertProperEnumeratorInitialization(args, options);
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
		PropertiesConfiguration options = ResourceTestUtils.generateTestingOptionsRequired("any");
		options.setProperty("ENUMERATION", "prune");

		Enumerator actualEnumerator = assertProperEnumeratorInitialization(args, options);
		Assert.assertTrue(actualEnumerator instanceof PruneBasedEnumerator);
	}

	@Test
	public void initInterestEnumeratorWithWithAllOptionsTest() {
		// set all the flags to opposite values to their defaults
		String[] args = {
				"-f", HOME+"Algorithm_L2SVM.dml",
		};
		PropertiesConfiguration options = ResourceTestUtils.generateTestingOptionsRequired("any");
		options.setProperty("ENUMERATION", "interest");
		options.setProperty("USE_LARGEST_ESTIMATE", "false");
		options.setProperty("USE_CP_ESTIMATES", "false");
		options.setProperty("USE_BROADCASTS", "false");
		options.setProperty("USE_OUTPUTS", "true");

		InterestBasedEnumerator actualEnumerator =
				(InterestBasedEnumerator) assertProperEnumeratorInitialization(args, options);
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
		PropertiesConfiguration options = ResourceTestUtils.generateTestingOptionsRequired("any");
		options.setProperty("INSTANCE_FAMILIES", "m5");
		options.setProperty("INSTANCE_SIZES", "2xlarge");

		Enumerator actualEnumerator = assertProperEnumeratorInitialization(args, options);

		HashMap<String, CloudInstance> inputInstances = ResourceTestUtils.getSimpleCloudInstanceMap();
		HashMap<String, CloudInstance> expectedInstances = new HashMap<>();
		expectedInstances.put("m5.2xlarge", inputInstances.get("m5.2xlarge"));

		HashMap<String, CloudInstance> actualInstances = actualEnumerator.getInstances();

		for (String instanceName: expectedInstances.keySet()) {
			ResourceTestUtils.assertEqualsCloudInstances(expectedInstances.get(instanceName), actualInstances.get(instanceName));
		}
	}

	@Test
	public void initEnumeratorWithCustomCPUQuotaTest() {
		String[] args = {
				"-f", HOME+"Algorithm_L2SVM.dml",
		};
		PropertiesConfiguration options = ResourceTestUtils.generateTestingOptionsRequired("any");
		options.setProperty("CPU_QUOTA", "256");

		Enumerator actualEnumerator = assertProperEnumeratorInitialization(args, options);

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
		Options cliOptions = createOptions();
		CommandLineParser clParser = new PosixParser();
		CommandLine line = null;
		try {
			line = clParser.parse(cliOptions, args);
		} catch (ParseException e) {
			Assert.fail("ParseException should not have been raise here: "+e);
		}
		PropertiesConfiguration options = ResourceTestUtils.generateTestingOptionsRequired(tmpOutFolder.toString());
		options.setProperty("MAX_EXECUTORS", "10");

		ResourceOptimizer.execute(line, options);

		if (!DEBUG) {
			ResourceTestUtils.deleteDirectoryWithFiles(tmpOutFolder);
		}
	}

	@Test
	public void executeForL2SVM_MinimalSearchSpace_C5_XLARGE_Test() throws IOException, ParseException {
		Path tmpOutFolder = Files.createTempDirectory("out");

		String[] args = {
				"-f", HOME+"Algorithm_L2SVM.dml",
				"-nvargs", "m=200000", "n=10000"
		};
		Options cliOptions = createOptions();
		CommandLineParser clParser = new PosixParser();
		CommandLine line = null;
		try {
			line = clParser.parse(cliOptions, args);
		} catch (ParseException e) {
			Assert.fail("ParseException should not have been raise here: "+e);
		}
		PropertiesConfiguration options = ResourceTestUtils.generateTestingOptionsRequired(tmpOutFolder.toString());
		options.setProperty("MAX_EXECUTORS", "10");
		options.setProperty("INSTANCE_FAMILIES", "c5,c5d,c5n");
		options.setProperty("INSTANCE_SIZES", "xlarge");

		ResourceOptimizer.execute(line, options);

		if (!DEBUG) {
			ResourceTestUtils.deleteDirectoryWithFiles(tmpOutFolder);
		}
	}

	@Test
	@Ignore //disabled dependencies
	public void executeForReadAndWrite_Test() throws IOException, ParseException {
		Path tmpOutFolder = Files.createTempDirectory("out");

		String[] args = {
				"-f", HOME+"ReadAndWrite.dml",
				"-nvargs",
					"fileA=s3://data/in/A.csv",
					"fileA_Csv=s3://data/out/A.csv",
					"fileA_Text=s3://data/out/A.txt"
		};
		Options cliOptions = createOptions();
		CommandLineParser clParser = new PosixParser();
		CommandLine line = null;
		try {
			line = clParser.parse(cliOptions, args);
		} catch (ParseException e) {
			Assert.fail("ParseException should not have been raise here: "+e);
		}
		PropertiesConfiguration options = ResourceTestUtils.generateTestingOptionsRequired(tmpOutFolder.toString());
		options.setProperty("MAX_EXECUTORS", "2");
		String localInputs = "s3://data/in/A.csv=" + HOME + "data/A.csv";
		options.setProperty("LOCAL_INPUTS", localInputs);

		ResourceOptimizer.execute(line, options);

		if (!DEBUG) {
			ResourceTestUtils.deleteDirectoryWithFiles(tmpOutFolder);
		}
	}

	// Helpers ---------------------------------------------------------------------------------------------------------

	private Enumerator assertProperEnumeratorInitialization(String[] args, PropertiesConfiguration options) {
		Options cliOptions = createOptions();
		CommandLineParser clParser = new PosixParser();
		CommandLine line = null;
		try {
			line = clParser.parse(cliOptions, args);
		} catch (ParseException e) {
			Assert.fail("ParseException should not have been raise here: "+e);
		}
		Enumerator actualEnumerator = null;
		try {
			actualEnumerator = initEnumerator(line, options);
		} catch (Exception e) {
			Assert.fail("Any exception should not have been raise here: "+e);
		}
		Assert.assertNotNull(actualEnumerator);

		return actualEnumerator;
	}
}
