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

package org.apache.sysds.test.functions.builtin.part2;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class BuiltinPCATest extends AutomatedTestBase {
	private final static String TEST_NAME = "pca";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinPCATest.class.getSimpleName() + "/";

	//Note: for <110 fine, but failing for more columns w/ eigen
	// org.apache.commons.math3.exception.MaxCountExceededException: illegal state: convergence failed
	private final static int rows = 3000;
	private final static int cols = 110;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"PC","V"}));
	}

	@Test
	public void testPca4Hybrid() {
		runPCATest(4, ExecMode.HYBRID);
	}
	
	@Test
	public void testPca16Hybrid() {
		runPCATest(16, ExecMode.HYBRID);
	}
	
	@Test
	public void testPca4Spark() {
		runPCATest(4, ExecMode.SPARK);
	}
	
	@Test
	public void testPca16Spark() {
		runPCATest(16, ExecMode.SPARK);
	}

	private void runPCATest(int k, ExecMode mode) {
		ExecMode modeOld = setExecMode(mode);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			List<String> proArgs = new ArrayList<>();
	
			proArgs.add("-args");
			proArgs.add(input("X"));
			proArgs.add(String.valueOf(k));
			proArgs.add(output("PC"));
			proArgs.add(output("V"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			double[][] X = TestUtils.round(getRandomMatrix(rows, cols, 1, 5, 1.0, 7));
			writeInputMatrixWithMTD("X", X, true);
	
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			MatrixCharacteristics mc = readDMLMetaDataFile("PC");
			Assert.assertEquals(rows, mc.getRows());
			Assert.assertEquals(k, mc.getCols());
		}
		finally {
			resetExecMode(modeOld);
		}
	}
}
