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

import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class CSVParametersTest extends CSVTestBase {

	private final static String TEST_NAME = "csvprop_test";
	private final static String TEST_CLASS_DIR = TEST_DIR + CSVParametersTest.class.getSimpleName() + "/";

	private final static int rows = 1200;
	private final static int cols = 100;
	private static double sparsity = 0.1;

	private boolean _header = false;
	private String _delim = ",";
	private boolean _sparse = true;

	public CSVParametersTest(boolean header, String delim, boolean sparse) {
		_header = header;
		_delim = delim;
		_sparse = sparse;
	}

	@Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][] {
			// header sep sparse
			{false, ",", true}, {false, ",", false}, {true, ",", true}, {true, ",", false}, {false, "|.", true},
			{false, "|.", false}, {true, "|.", true}, {true, "|.", false}};

		return Arrays.asList(data);
	}

	private void setup() {

		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("w_header", _header);
		config.addVariable("w_delim", _delim);
		config.addVariable("w_sparse", _sparse);

		loadTestConfiguration(config);
	}

	@Test
	public void testCSVParametersSparseCP() {
		setup();
		sparsity = 0.1;

		ExecMode old_platform = rtplatform;

		rtplatform = ExecMode.SINGLE_NODE;
		csvParameterTest(rtplatform, sparsity);

		rtplatform = old_platform;
	}

	@Test
	public void testCSVParametersDenseCP() {
		setup();
		sparsity = 1.0;

		ExecMode old_platform = rtplatform;

		rtplatform = ExecMode.SINGLE_NODE;
		csvParameterTest(rtplatform, sparsity);

		rtplatform = old_platform;
	}

	@Test
	public void testCSVParametersSparseHybrid() {
		setup();
		sparsity = 0.1;

		ExecMode old_platform = rtplatform;

		rtplatform = ExecMode.HYBRID;
		csvParameterTest(rtplatform, sparsity);

		rtplatform = old_platform;
	}

	@Test
	public void testCSVParametersDenseHybrid() {
		setup();
		sparsity = 1.0;

		ExecMode old_platform = rtplatform;

		rtplatform = ExecMode.HYBRID;
		csvParameterTest(rtplatform, sparsity);

		rtplatform = old_platform;
	}

	private void csvParameterTest(ExecMode platform, double sp) {

		// generate actual dataset
		double[][] D = getRandomMatrix(rows, cols, 0, 1, sp, 7777);
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, -1, -1);
		writeInputMatrixWithMTD("D", D, true, mc);
		D = null;

		String HOME = SCRIPT_DIR + TEST_DIR;
		String txtFile = input("D");
		// String binFile = input("D.binary");
		String csvFile = output("D.csv");
		String scalarFile = output("diff.scalar");

		String writeDML = HOME + "csvprop_write.dml";
		String[] writeArgs = new String[] {"-args", txtFile, csvFile, Boolean.toString(_header).toUpperCase(), _delim,
			Boolean.toString(_sparse).toUpperCase()};

		String readDML = HOME + "csvprop_read.dml";
		String[] readArgs = new String[] {"-args", txtFile, csvFile, Boolean.toString(_header).toUpperCase(), _delim,
			Boolean.toString(_sparse).toUpperCase(), Double.toString(0.0), scalarFile};

		// System.out.println("Text -> CSV");
		// Text -> CSV
		fullDMLScriptName = writeDML;
		programArgs = writeArgs;
		runTest(true, false, null, -1);

		// Evaluate the written CSV file
		// System.out.println("CSV -> SCALAR");
		fullDMLScriptName = readDML;
		programArgs = readArgs;
		// boolean exceptionExpected = (!_sparse && sparsity < 1.0);
		runTest(true, false, null, -1);

		double dmlScalar = TestUtils.readDMLScalar(scalarFile);
		TestUtils.compareScalars(dmlScalar, 0.0, eps);
	}

	protected String getTestName() {
		return TEST_NAME;
	}

	protected String getTestClassDir() {
		return TEST_CLASS_DIR;
	}
}
