/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.test.integration.applications;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

import org.junit.runners.Parameterized.Parameters;

import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.utils.TestUtils;

public abstract class ArimaTest extends AutomatedTestBase {
	
	protected final static String TEST_DIR = "applications/arima_box-jenkins/";
	protected final static String TEST_NAME = "arima";
	
	protected int max_func_invoc, p, d, q, P, D, Q, s, include_mean, useJacobi;
	
	public ArimaTest(int m, int p, int d, int q, int P, int D, int Q, int s, int include_mean, int useJacobi){
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
		Object[][] data = new Object[][] { //{ 10, 1, 0, 1, 0, 0, 0, 24, 0, 0}, 
										   { 10, 1, 1, 1, 1, 1, 1, 24, 1, 1}};
		return Arrays.asList(data);
	}
	
	@Override
	public void setUp() {
    	addTestConfiguration(TEST_DIR, TEST_NAME);
	}
	
	protected void testArima(ScriptType scriptType) {
		System.out.println("------------ BEGIN " + TEST_NAME + " " + scriptType + " TEST WITH {" +
			max_func_invoc + ", " + 
			p + ", " + 
			d + ", " + 
			q + ", " + 
			P + ", " + 
			D + ", " + 
			Q + ", " + 
			s + ", " + 
			include_mean + ", " + 
			useJacobi+ "} ------------");
		this.scriptType = scriptType;
		
		getAndLoadTestConfiguration(TEST_NAME);
	
		List<String> proArgs = new ArrayList<String>();
		if (scriptType == ScriptType.PYDML) {
			proArgs.add("-python");
		}
		proArgs.add("-args");
		proArgs.add(input("col.mtx"));
		proArgs.add(Integer.toString(max_func_invoc));
		proArgs.add(Integer.toString(p));
		proArgs.add(Integer.toString(d));
		proArgs.add(Integer.toString(q));
		proArgs.add(Integer.toString(P));
		proArgs.add(Integer.toString(D));
		proArgs.add(Integer.toString(Q));
		proArgs.add(Integer.toString(s));
		proArgs.add(Integer.toString(include_mean));
		proArgs.add(Integer.toString(useJacobi));
		proArgs.add(output("learnt.model"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		System.out.println("arguments from test case: " + Arrays.toString(programArgs));
		
		fullDMLScriptName = getScript();

		rCmd = getRCmd(inputDir(), Integer.toString(max_func_invoc), Integer.toString(p), Integer.toString(d), Integer.toString(q), Integer.toString(P), 
				Integer.toString(D), Integer.toString(Q), Integer.toString(s), Integer.toString(include_mean), Integer.toString(useJacobi), expectedDir());
		
		int timeSeriesLength = 5000;
		double[][] timeSeries = getRandomMatrix(timeSeriesLength, 1, 1, 5, 0.9, System.currentTimeMillis());
		
		MatrixCharacteristics mc = new MatrixCharacteristics(timeSeriesLength,1,-1,-1);
		writeInputMatrixWithMTD("col", timeSeries, true, mc);
		
		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
		
		runRScript(true);

		double tol = Math.pow(10, -14);
		HashMap<CellIndex, Double> arima_model_R = readRMatrixFromFS("learnt.model");
        HashMap<CellIndex, Double> arima_model_SYSTEMML= readDMLMatrixFromHDFS("learnt.model");
        TestUtils.compareMatrices(arima_model_R, arima_model_SYSTEMML, tol, "arima_model_R", "arima_model_SYSTEMML");
	}   
}
