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

package org.apache.sysds.test.functions.builtin.part1;

import java.util.HashMap;

import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Assert;
import org.junit.Test;

public class BuiltinCoresetDTTest extends AutomatedTestBase {

	private final static String TEST_NAME = "coresetDT";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinCoresetDTTest.class.getSimpleName() + "/";

	private final static String WINE_DATA = DATASET_DIR + "wine/winequality-red-white.csv";
	private final static String WINE_TFSPEC = DATASET_DIR + "wine/tfspec.json";

	// Test accuracy % drop of the coreset trained model vs the full data model
	private static final double LOGREG_ACC_DROP_TOLE = 2.0;
	private static final double DTREE_ACC_DROP_TOLE  = 7.0;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"acc"}));
	}

	@Test public void testWine10CP() { runCoresetDT(0.10); }
	@Test public void testWine20CP() { runCoresetDT(0.20); }
	@Test public void testWine50CP() { runCoresetDT(0.50); }

	private void runCoresetDT(double fraction) {
		setExecMode(ExecType.CP);
		loadTestConfiguration(getTestConfiguration(TEST_NAME));

		fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", WINE_DATA, WINE_TFSPEC, Double.toString(fraction), output("acc")};
		runTest(true, false, null, -1);

		HashMap<CellIndex, Double> acc = readDMLMatrixFromOutputDir("acc");
		check("logreg", acc.get(new CellIndex(1,1)), acc.get(new CellIndex(1,2)), LOGREG_ACC_DROP_TOLE);
		check("dtree",  acc.get(new CellIndex(2,1)), acc.get(new CellIndex(2,2)), DTREE_ACC_DROP_TOLE);
	}

	private void check(String name, double full, double core, double tol) {
		Assert.assertTrue(name + " coreset accuracy " + core + "% dropped more than " + tol	+ "pp below full accuracy " + full + "%", core >= full - tol);
	}
}
