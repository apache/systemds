/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.test.applications;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestUtils;

@RunWith(value = Parameterized.class)
public class ArimaTest extends AutomatedTestBase {
	
	protected final static String TEST_DIR = "applications/arima_box-jenkins/";
	protected final static String TEST_NAME1 = "arima";
	protected final static String TEST_NAME2 = "arima_old";
	
	protected String TEST_CLASS_DIR = TEST_DIR + ArimaTest.class.getSimpleName() + "/";
	
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
		return Arrays.asList(new Object[][] {
			{10, 1, 1, 1, 1, 1, 1, 24, 1, 1}});
			//TODO include after ARIMA script modifications
			//(these tests are currently failing due to invalid loop ranges)
			//{0, 7, 0, 0, 0, 0, 0, 0, 0, 0},   //AR(7)
			//{0, 0, 0, 3, 0, 0, 0, 0, 0, 0}}); //MA(3)
	}
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_CLASS_DIR, TEST_NAME1);
		addTestConfiguration(TEST_CLASS_DIR, TEST_NAME2);
	}
	
	@Test
	public void testArima() {
		System.out.println("------------ BEGIN " + TEST_NAME1 + " TEST WITH {" +
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
		
		List<String> proArgs = new ArrayList<String>();
		getAndLoadTestConfiguration(TEST_NAME2);
		
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
		
		
		/* TODO use after R script is made consistent 
			getAndLoadTestConfiguration(TEST_NAME2);
			proArgs.add("-nvargs");
			proArgs.add("X="+input("col.mtx"));
			proArgs.add("max_func="+Integer.toString(max_func_invoc));
			proArgs.add("p="+Integer.toString(p));
			proArgs.add("d="+Integer.toString(d));
			proArgs.add("q="+Integer.toString(q));
			proArgs.add("P="+Integer.toString(P));
			proArgs.add("D="+Integer.toString(D));
			proArgs.add("Q="+Integer.toString(Q));
			proArgs.add("s="+Integer.toString(s));
			proArgs.add("include_mean="+Integer.toString(include_mean));
			proArgs.add("solver="+(useJacobi==1?"jacobi":"cg_solver"));
			proArgs.add("dest="+output("learnt.model"));
		*/
		
		programArgs = proArgs.toArray(new String[proArgs.size()]);
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
		HashMap<CellIndex, Double> arima_model_SYSTEMDS= readDMLMatrixFromHDFS("learnt.model");
		TestUtils.compareMatrices(arima_model_R, arima_model_SYSTEMDS, tol, "arima_model_R", "arima_model_SYSTEMDS");
	}
}
