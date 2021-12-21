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

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import org.junit.Test;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.functions.builtin.part1.BuiltinArimaTest;

import java.util.concurrent.ThreadLocalRandom;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class BuiltinOutlierByArima extends AutomatedTestBase {
	private final static String TEST_NAME = "outlierByArima";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinArimaTest.class.getSimpleName() + "/";

	protected int max_func_invoc, p, d, q, P, D, Q, s, include_mean, useJacobi, repairMethod;

	public BuiltinOutlierByArima(int m, int p, int d, int q, int P, int D, int Q, int s, int include_mean, int useJacobi, int repairMethod){
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
		this.repairMethod = repairMethod;
	}

	@Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			{1, 2, 0, 0, 0, 0, 0, 24, 1, 1, 1},
			{1, 2, 0, 0, 0, 0, 0, 24, 1, 1, 2},
			{1, 2, 0, 0, 0, 0, 0, 24, 1, 1, 3}});
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"}));
	}

	@Test
	public void testOutlierByArima(){
		Types.ExecMode platformOld = setExecMode(ExecType.CP);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			fullRScriptName = HOME + TEST_NAME + ".R";

			programArgs = new String[]{
				"-nvargs", "X=" + input("col.mtx"), "p=" + p, "repairMethod=" + 1,
				"outputfilename=" + output("result"),};
			rCmd = getRCmd(input("bad.mtx"), expected("result"));

			int timeSeriesLength = 3000;
			int num_outliers = 10;
			double[][] timeSeries = getRandomMatrix(timeSeriesLength, 1, 1, 3, 1, System.currentTimeMillis());
			double[][] comparisonSeries = deepCopy(timeSeries);
			for(int i=0; i<num_outliers; i++) {
				int r = ThreadLocalRandom.current().nextInt(0, timeSeries.length);
				double badValue = ThreadLocalRandom.current().nextDouble(10, 50);
				timeSeries[r][0] = badValue;
				if (repairMethod == 1)
					comparisonSeries[r][0] = 0.0;
				else if (repairMethod == 2)
					comparisonSeries[r][0] = Double.NaN;
			}

			MatrixCharacteristics mc = new MatrixCharacteristics(timeSeriesLength,1,-1,-1);
			writeInputMatrixWithMTD("col", timeSeries, true, mc);
			writeInputMatrixWithMTD("bad", comparisonSeries, true, mc);

			runTest(true, false, null, -1);
			runRScript(true);

			HashMap<CellIndex, Double> time_series_SYSTEMDS = readDMLMatrixFromOutputDir("result");
			HashMap<CellIndex, Double> time_series_real = readRMatrixFromExpectedDir("result");

			double tol = Math.pow(10, -12);
			if (repairMethod == 3)
				TestUtils.compareScalars(time_series_real.size()-num_outliers, time_series_SYSTEMDS.size(), tol);
			else
				TestUtils.compareMatrices(time_series_real, time_series_SYSTEMDS, tol, "time_series_real", "time_series_SYSTEMDS");
		}
		finally {
			rtplatform = platformOld;
		}
	}

	private static double[][] deepCopy(double[][] input) {
		double[][] result = new double[input.length][];
		for (int r = 0; r < input.length; r++) {
			result[r] = input[r].clone();
		}
		return result;
	}
}
