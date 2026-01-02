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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(Parameterized.class)
public class ReadHDF5Test extends ReadHDF5TestBase {

	private static final double eps = 1e-9;
	private static final String TEST_NAME = "ReadHDF5Test";

	private static final int S_2D_ROWS = 200;
	private static final int S_2D_COLS = 40;
	private static final int S_ARRAY_LENGTH = 30;
	private static final int MATRIX_3D_ROWS = 15;
	private static final int MATRIX_3D_FLATTENED_COLS = 15 * 5;
	private static final int MULTI_TENSOR_SAMPLES = 120;
	private static final int MULTI_TENSOR_LABEL_FEATURES = 12;
	private static final int MULTI_TENSOR_SEN1_FLATTENED_COLS = 16 * 16 * 4;

	private static final List<Hdf5TestCase> TEST_CASES = Collections.unmodifiableList(Arrays.asList(
		new Hdf5TestCase(
			"test_single_dataset.h5", "data", DmlVariant.FORMAT_AND_DATASET, S_2D_ROWS, S_2D_COLS),
		new Hdf5TestCase(
			"test_multiple_datasets.h5", "matrix_2d", DmlVariant.DATASET_ONLY, S_2D_ROWS, S_2D_COLS),
		new Hdf5TestCase(
			"test_multiple_datasets.h5", "matrix_3d", DmlVariant.DATASET_ONLY, MATRIX_3D_ROWS, MATRIX_3D_FLATTENED_COLS),
		new Hdf5TestCase(
			"test_different_dtypes.h5", "double_primary", DmlVariant.DATASET_ONLY, S_2D_ROWS, S_2D_COLS),
		new Hdf5TestCase(
			"test_chunked.h5", "chunked_data", DmlVariant.FORMAT_AND_DATASET, S_2D_ROWS, S_2D_COLS),
		new Hdf5TestCase(
			"test_compressed.h5", "gzip_compressed_9", DmlVariant.DATASET_ONLY, S_2D_ROWS, S_2D_COLS),
		new Hdf5TestCase(
			"test_multi_tensor_samples.h5", "label", DmlVariant.DATASET_ONLY, MULTI_TENSOR_SAMPLES, MULTI_TENSOR_LABEL_FEATURES),
		new Hdf5TestCase(
			"test_multi_tensor_samples.h5", "sen1", DmlVariant.DATASET_ONLY, MULTI_TENSOR_SAMPLES, MULTI_TENSOR_SEN1_FLATTENED_COLS),
		new Hdf5TestCase(
			"test_nested_groups.h5", "group1/subgroup/data2", DmlVariant.FORMAT_AND_DATASET, S_2D_ROWS, S_2D_COLS),
		new Hdf5TestCase(
			"test_with_attributes.h5", "data", DmlVariant.DATASET_ONLY, S_2D_ROWS, S_2D_COLS),
		new Hdf5TestCase(
			"test_empty_datasets.h5", "empty", DmlVariant.FORMAT_AND_DATASET, 0, S_2D_COLS),
		new Hdf5TestCase(
			"test_string_datasets.h5", "string_array", DmlVariant.DATASET_ONLY, S_ARRAY_LENGTH, 1)
	));

	private final Hdf5TestCase testCase;

	public ReadHDF5Test(Hdf5TestCase testCase) {
		this.testCase = testCase;
	}

	@BeforeClass
	public static void ensureHdf5DataGenerated() {
		Path scriptDir = Paths.get(SCRIPT_DIR, TEST_DIR);
		Path inputDir = scriptDir.resolve(INPUT_DIR);
		boolean missingFiles = TEST_CASES.stream()
			.anyMatch(tc -> Files.notExists(inputDir.resolve(tc.hdf5File)));
		if(!missingFiles)
			ensureMetadataFiles(inputDir);
		else {
			generateHdf5Data(scriptDir);

			boolean stillMissing = TEST_CASES.stream()
				.anyMatch(tc -> Files.notExists(inputDir.resolve(tc.hdf5File)));
			if(stillMissing)
				Assert.fail("Failed to generate required HDF5 files for ReadHDF5 tests.");

			ensureMetadataFiles(inputDir);
		}
	}

	@Parameters(name = "{0}")
	public static Collection<Object[]> data() {
		return TEST_CASES.stream()
			.map(tc -> new Object[] {tc})
			.collect(Collectors.toList());
	}

	@Override
	protected String getTestName() {
		return TEST_NAME;
	}

	@Override
	protected String getTestClassDir() {
		return TEST_CLASS_DIR;
	}

	@Test
	public void testReadSequential() {
		runReadHDF5Test(testCase, ExecMode.SINGLE_NODE, false);
	}

	@Test
	public void testReadSequentialParallelIO() {
		runReadHDF5Test(testCase, ExecMode.SINGLE_NODE, true);
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

			fullRScriptName = HOME + "ReadHDF5_Verify.R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputMatrixName + " " + testCase.dataset + " " + expectedDir();

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

	private static void ensureMetadataFiles(Path inputDir) {
		try {
			Files.createDirectories(inputDir);
			for(Hdf5TestCase tc : TEST_CASES) {
				Path mtdPath = inputDir.resolve(tc.getMtdFileName());
				if(Files.exists(mtdPath))
					continue;

				MatrixCharacteristics mc = new MatrixCharacteristics(tc.rows, tc.cols, tc.getNonZeros());
				HDFSTool.writeMetaDataFile(mtdPath.toString(), ValueType.FP64, mc, FileFormat.HDF5);
			}
		}
		catch(IOException e) {
			Assert.fail("Unable to create HDF5 metadata files: " + e.getMessage());
		}
	}

	private enum DmlVariant {
		FORMAT_AND_DATASET("ReadHDF5_WithFormatAndDataset.dml"),
		DATASET_ONLY("ReadHDF5_WithDataset.dml"),
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
		private final long rows;
		private final long cols;

		private Hdf5TestCase(String hdf5File, String dataset, DmlVariant variant, long rows, long cols) {
			this.hdf5File = hdf5File;
			this.dataset = dataset;
			this.variant = variant;
			this.rows = rows;
			this.cols = cols;
		}

		private String getMtdFileName() {
			return hdf5File + ".mtd";
		}

		private long getNonZeros() {
			if(rows == 0 || cols == 0)
				return 0;
			return rows * cols;
		}

		@Override
		public String toString() {
			return hdf5File + "::" + dataset;
		}
	}
}
