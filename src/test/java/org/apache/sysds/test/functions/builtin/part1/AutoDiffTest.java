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

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class AutoDiffTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "autoDiff";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + AutoDiffTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}

	@Test
	public void testAutoDiffCP1() {
		runAutoDiffTest(Types.ExecType.CP);
	}

	private void runAutoDiffTest(Types.ExecType instType)
	{
		ExecMode platformOld = setExecMode(instType);

		try
		{
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = false;
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-lineage", "-args", output("dX"), output("ad_dX")};
			runTest(true, false, null, -1);
			HashMap<MatrixValue.CellIndex, Double> dml_dX = readDMLMatrixFromOutputDir("dX");
			HashMap<MatrixValue.CellIndex, Double> autoDiff_dX = readDMLMatrixFromOutputDir("ad_dX");
			TestUtils.compareMatrices(dml_dX, autoDiff_dX, 1e-6, "Stat-DML", "Stat-AutoDiff");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
