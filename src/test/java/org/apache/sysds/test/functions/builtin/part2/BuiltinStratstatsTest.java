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
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class BuiltinStratstatsTest extends AutomatedTestBase {
	private final static String TEST_NAME = "stratstats";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinStratstatsTest.class.getSimpleName() + "/";

	protected String X;
	protected int rows, cols;

	public BuiltinStratstatsTest(String X, int cols, int rows){
		this.X = X;
		this.cols = cols;
		this.rows = rows;
	}

	@Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			{"randfile", 10, 6},
			{"randfile", 4, 18}});
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"}));
	}

	@Test
	public void testStratstats(){
		Types.ExecMode platformOld = setExecMode(ExecType.CP);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			fullRScriptName = HOME + TEST_NAME + ".R";

			programArgs = new String[]{
					"-nvargs", "X=" + input("random.mtx"), "O=" + output("result"), "Y=" + input("doesnotexist"),};
			rCmd = getRCmd(input("random.mtx"), expected("result"));

			double[][] values = getRandomMatrix(rows, cols, 10,40, 1, 54321);
			MatrixCharacteristics mc = new MatrixCharacteristics(rows,cols,-1,-1);
			writeInputMatrixWithMTD("random", values, true, mc);

			runTest(true, false, null, -1);
			runRScript(true);

			HashMap<CellIndex, Double> stats_SYSTEMDS = readDMLMatrixFromOutputDir("result");
			HashMap<CellIndex, Double> stats_real = readRMatrixFromExpectedDir("result");

			double tol = Math.pow(10, -3);
			TestUtils.compareMatrices(stats_real, stats_SYSTEMDS, tol, "stats_real", "stats_SYSTEMDS", true);
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
