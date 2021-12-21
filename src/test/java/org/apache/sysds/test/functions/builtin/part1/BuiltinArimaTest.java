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

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class BuiltinArimaTest extends AutomatedTestBase {
	private final static String TEST_NAME = "arima";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinArimaTest.class.getSimpleName() + "/";

	protected int max_func_invoc, p, d, q, P, D, Q, s, include_mean, useJacobi;

	public BuiltinArimaTest(int m, int p, int d, int q, int P, int D, int Q, int s, int include_mean, int useJacobi){
		this.max_func_invoc = m;
		this.p = p;
		this.d = d;
		this.q = q;
		this.P = P;
		this.D = D;
		this.Q = Q;
		this.s = s;
		this.include_mean = include_mean;
		this.useJacobi = useJacobi;
	}

	@Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
				{20, 1, 0, 0, 0, 0, 0, 24, 1, 1},
				{20, 0, 0, 1, 0, 0, 0, 24, 1, 1},
				{20, 2, 0, 1, 0, 0, 0, 24, 1, 1},
				
				//TODO fix remaining configurations (e.g., differencing)
				//{10, 1, 0, 10, 0, 0, 0, 24, 1, 1}
				// {10, 1, 1, 2, 0, 0, 0, 24, 1, 1},
				// {10, 0, 1, 2, 0, 0, 0, 24, 1, 1},
				// {10, 0, 0, 0, 1, 1, 0, 24, 1, 1},
				// {10, 0, 0, 0, 1, 1, 2, 24, 1, 1},
				// {10, 0, 0, 0, 0, 1, 2, 24, 1, 1}}
		});
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"}));
	}

	@Test
	public void testArima(){
		ExecMode platformOld = setExecMode(ExecMode.HYBRID);
		
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			fullRScriptName = HOME + TEST_NAME + ".R";

			programArgs = new String[]{
					"-nvargs", "X=" + input("col.mtx"), "max_func_invoc=" + max_func_invoc,
					"p=" + p, "d=" + d, "q=" + q, "P=" + P, "D=" + D, "Q=" + Q,
					"s=" + s, "include_mean=" + include_mean, "useJacobi=" + useJacobi,
					"model=" + output("learnt.model"),};

			rCmd = getRCmd(input("col.mtx"), Integer.toString(max_func_invoc), Integer.toString(p),
				Integer.toString(d), Integer.toString(q), Integer.toString(P), Integer.toString(D),
				Integer.toString(Q), Integer.toString(s), Integer.toString(include_mean),
				Integer.toString(useJacobi), expected("learnt.model"));

			int timeSeriesLength = 3000;
			double[][] timeSeries = getRandomMatrix(timeSeriesLength, 1, 1, 5, 0.9, 54321);

			MatrixCharacteristics mc = new MatrixCharacteristics(timeSeriesLength,1,-1,-1);
			writeInputMatrixWithMTD("col", timeSeries, true, mc);

			runTest(true, false, null, -1);
			runRScript(true);

			double tol = Math.pow(10, -14);
			HashMap<CellIndex, Double> arima_model_R = readRMatrixFromExpectedDir("learnt.model");
			HashMap<CellIndex, Double> arima_model_SYSTEMDS= readDMLMatrixFromOutputDir("learnt.model");
			TestUtils.compareMatrices(arima_model_R, arima_model_SYSTEMDS, tol, "arima_R", "arima_SYSTEMDS");
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
