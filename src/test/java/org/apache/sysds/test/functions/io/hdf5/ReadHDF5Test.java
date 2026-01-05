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

package org.apache.sysds.test.functions.io.hdf5;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.io.File;
import org.apache.commons.io.FileUtils;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

public class ReadHDF5Test extends ReadHDF5TestBase {

	private static final double eps = 1e-9;
	private static final String TEST_NAME = "ReadHDF5Test";

	private static final List<Hdf5TestCase> TEST_CASES = Collections.unmodifiableList(
		Arrays.asList(new Hdf5TestCase("test_single_dataset.h5", "data", DmlVariant.FORMAT_AND_DATASET),
			new Hdf5TestCase("test_multiple_datasets.h5", "matrix_2d", DmlVariant.DATASET_ONLY),
			new Hdf5TestCase("test_multiple_datasets.h5", "matrix_3d", DmlVariant.DATASET_ONLY),
			new Hdf5TestCase("test_multi_tensor_samples.h5", "label", DmlVariant.DATASET_ONLY),
			new Hdf5TestCase("test_multi_tensor_samples.h5", "sen1", DmlVariant.DATASET_ONLY),
			new Hdf5TestCase("test_nested_groups.h5", "group1/subgroup/data2", DmlVariant.FORMAT_AND_DATASET)));

	@Override
	protected String getTestName() {
		return TEST_NAME;
	}

	@Override
	protected String getTestClassDir() {
		return TEST_CLASS_DIR;
	}

	@BeforeClass
	public static void setUpClass() {
		Path scriptDir = Paths.get(SCRIPT_DIR + TEST_DIR);
		generateHdf5Data(scriptDir);
	}

	@Test
	public void testReadSequential() {
		for(Hdf5TestCase tc : TEST_CASES)
			runReadHDF5Test(tc, ExecMode.SINGLE_NODE, false);
	}

	@Test
	public void testReadSequentialParallelIO() {
		for(Hdf5TestCase tc : TEST_CASES)
			runReadHDF5Test(tc, ExecMode.SINGLE_NODE, true);
	}

	protected void runReadHDF5Test(Hdf5TestCase testCase, ExecMode platform, boolean parallel) {
		ExecMode oldPlatform = rtplatform;
		rtplatform = platform;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if(rtplatform == ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		boolean oldpar = CompilerConfig.FLAG_PARREADWRITE_TEXT;

		try {
			CompilerConfig.FLAG_PARREADWRITE_TEXT = parallel;

			TestConfiguration config = getTestConfiguration(getTestName());
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			String inputMatrixName = HOME + INPUT_DIR + testCase.hdf5File;

			fullDMLScriptName = HOME + testCase.variant.getScriptName();
			programArgs = new String[] {"-args", inputMatrixName, testCase.dataset, output("Y")};

			// Clean per-case output/expected to avoid reusing stale metadata between looped cases
			String outY = output("Y");
			String expY = expected("Y");
			FileUtils.deleteQuietly(new File(outY));
			FileUtils.deleteQuietly(new File(outY + ".mtd"));
			FileUtils.deleteQuietly(new File(expY));
			FileUtils.deleteQuietly(new File(expY + ".mtd"));

			fullRScriptName = HOME + "ReadHDF5_Verify.R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputMatrixName + " " + testCase.dataset + " "
				+ expectedDir();

			runTest(true, false, null, -1);
			runRScript(true);

			HashMap<MatrixValue.CellIndex, Double> YR = readRMatrixFromExpectedDir("Y");
			HashMap<MatrixValue.CellIndex, Double> YSYSTEMDS = readDMLMatrixFromOutputDir("Y");
			TestUtils.compareMatrices(YR, YSYSTEMDS, eps, "YR", "YSYSTEMDS");
		}
		finally {
			rtplatform = oldPlatform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

	private static void generateHdf5Data(Path scriptDir) {
		ProcessBuilder processBuilder = new ProcessBuilder("Rscript", "gen_HDF5_testdata.R");
		processBuilder.directory(scriptDir.toFile());
		processBuilder.redirectErrorStream(true);

		try {
			Process process = processBuilder.start();
			StringBuilder output = new StringBuilder();
			try(BufferedReader reader = new BufferedReader(
				new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
				reader.lines().forEach(line -> output.append(line).append(System.lineSeparator()));
			}
			int exitCode = process.waitFor();
			if(exitCode != 0)
				Assert.fail("Failed to execute gen_HDF5_testdata.R (exit " + exitCode + "):\n" + output);
		}
		catch(IOException e) {
			Assert.fail("Unable to execute gen_HDF5_testdata.R: " + e.getMessage());
		}
		catch(InterruptedException e) {
			Thread.currentThread().interrupt();
			Assert.fail("Interrupted while generating HDF5 test data.");
		}
	}

	private enum DmlVariant {
		FORMAT_AND_DATASET("ReadHDF5_WithFormatAndDataset.dml"), DATASET_ONLY("ReadHDF5_WithDataset.dml"),
		DEFAULT("ReadHDF5_Default.dml");

		private final String scriptName;

		DmlVariant(String scriptName) {
			this.scriptName = scriptName;
		}

		public String getScriptName() {
			return scriptName;
		}
	}

	private static final class Hdf5TestCase {
		private final String hdf5File;
		private final String dataset;
		private final DmlVariant variant;

		private Hdf5TestCase(String hdf5File, String dataset, DmlVariant variant) {
			this.hdf5File = hdf5File;
			this.dataset = dataset;
			this.variant = variant;
		}

		@Override
		public String toString() {
			return hdf5File + "::" + dataset;
		}
	}
}
