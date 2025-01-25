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

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

public class BuiltinSliceLineRealDataTest extends AutomatedTestBase {
	private final static String TEST_NAME = "sliceLineRealData";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinSliceLineRealDataTest.class.getSimpleName() + "/";

	private final static String SALARIES_DATA = DATASET_DIR + "Salaries.csv";
	private final static String SALARIES_TFSPEC = DATASET_DIR + "Salaries_tfspec.json";

	@Override
	public void setUp() {
		for(int i=1; i<=1; i++)
			addTestConfiguration(TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R","V"}));
	}

	@Test
	public void testSliceLineSalaries() {
		runSliceLine(1, SALARIES_DATA, SALARIES_TFSPEC, 0.5, ExecType.CP);
	}

	private void runSliceLine(int test, String data, String tfspec, double minAcc, ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME+test));
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats",
				"-args", data, tfspec, output("R"), output("V")};

			runTest(true, false, null, -1);

			double acc = readDMLMatrixFromOutputDir("R").get(new CellIndex(1,1));
			double val = readDMLMatrixFromOutputDir("V").get(new CellIndex(1,1));
			Assert.assertTrue(acc >= minAcc);
			Assert.assertTrue(val >= 0.99);
			Assert.assertEquals(0, Statistics.getNoOfExecutedSPInst());
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
