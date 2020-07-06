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

package org.apache.sysds.test.functions.io.csv;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public abstract class FormatChangeTest extends CSVTestBase {

	private int _rows, _cols;
	private double _sparsity;

	public FormatChangeTest(int r, int c, double sp) {
		_rows = r;
		_cols = c;
		_sparsity = sp;
	}

	private void setup() {
		TestConfiguration config = getTestConfiguration(getTestName());
		config.addVariable("rows", _rows);
		config.addVariable("cols", _cols);
		config.addVariable("format1", "text");
		config.addVariable("format2", "binary");
		loadTestConfiguration(config);
	}

	@Test
	public void testFormatChangeCP() {
		setup();
		ExecMode old_platform = rtplatform;
		rtplatform = ExecMode.SINGLE_NODE;
		formatChangeTest();
		rtplatform = old_platform;
	}

	@Test
	public void testFormatChangeHybrid() {
		setup();
		ExecMode old_platform = rtplatform;
		rtplatform = ExecMode.HYBRID;
		formatChangeTest();
		rtplatform = old_platform;
	}

	private void formatChangeTest() {
		int rows = _rows;
		int cols = _cols;
		double sparsity = _sparsity;

		// generate actual dataset
		double[][] D = getRandomMatrix(rows, cols, 0, 1, sparsity, 7777);
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, -1, -1);
		writeInputMatrixWithMTD("D", D, true, mc);

		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + "csv_test.dml";
		String[] oldProgramArgs = programArgs = new String[] {"-args", input("D"), null, input("D.binary"), null};

		String txtFile = input("D");
		String binFile = input("D.binary");
		String csvFile = output("D.csv");

		// text to binary format
		programArgs[2] = "text";
		programArgs[3] = binFile;
		programArgs[4] = "binary";
		runTest(true, false, null, -1);

		// Test TextCell -> CSV conversion
		LOG.debug("TextCell -> CSV");
		programArgs[2] = "text";
		programArgs[3] = csvFile;
		programArgs[4] = "csv";
		runTest(true, false, null, -1);

		compareFiles(rows, cols, sparsity, txtFile, "text", csvFile);

		// Test BinaryBlock -> CSV conversion
		LOG.debug("BinaryBlock -> CSV");
		programArgs = oldProgramArgs;
		programArgs[1] = binFile;
		programArgs[2] = "binary";
		programArgs[3] = csvFile;
		programArgs[4] = "csv";
		runTest(true, false, null, -1);

		compareFiles(rows, cols, sparsity, binFile, "binary", csvFile);

		// Test CSV -> TextCell conversion
		LOG.debug("CSV -> TextCell");
		programArgs = oldProgramArgs;
		programArgs[1] = csvFile;
		programArgs[2] = "csv";
		programArgs[3] = txtFile;
		programArgs[4] = "text";
		runTest(true, false, null, -1);

		compareFiles(rows, cols, sparsity, txtFile, "text", csvFile);

		// Test CSV -> BinaryBlock conversion
		LOG.debug("CSV -> BinaryBlock");
		programArgs = oldProgramArgs;
		programArgs[1] = csvFile;
		programArgs[2] = "csv";
		programArgs[3] = binFile;
		programArgs[4] = "binary";
		runTest(true, false, null, -1);

		compareFiles(rows, cols, sparsity, binFile, "binary", csvFile);

	}

	private void compareFiles(int rows, int cols, double sparsity, String dmlFile, String dmlFormat, String csvFile) {
		String HOME = SCRIPT_DIR + TEST_DIR;

		// backup old DML and R script files
		String oldDMLScript = fullDMLScriptName;
		String oldRScript = fullRScriptName;

		String dmlOutput = output("dml.scalar");
		String rOutput = output("R.scalar");

		fullDMLScriptName = HOME + "csv_verify.dml";
		programArgs = new String[] {"-args", dmlFile, Integer.toString(rows), Integer.toString(cols), dmlFormat,
			dmlOutput};

		// Check if input csvFile is a directory
		try {
			csvFile = TestUtils.processMultiPartCSVForR(csvFile);
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}

		fullRScriptName = HOME + "csv_verify.R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + csvFile + " " + rOutput;

		// Run the verify test
		runTest(true, false, null, -1);
		runRScript(true);

		double dmlScalar = TestUtils.readDMLScalar(dmlOutput);
		double rScalar = TestUtils.readRScalar(rOutput);

		TestUtils.compareScalars(dmlScalar, rScalar, eps);

		// restore old DML and R script files
		fullDMLScriptName = oldDMLScript;
		fullRScriptName = oldRScript;
	}
}
