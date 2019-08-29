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
public class MDABivariateStatsTest extends AutomatedTestBase 
{

	protected final static String TEST_DIR = "applications/mdabivar/";
	protected final static String TEST_NAME = "MDABivariateStats";
	protected String TEST_CLASS_DIR = TEST_DIR + MDABivariateStatsTest.class.getSimpleName() + "/";
	
	protected int n, m, label_index, label_measurement_level;
	
	public MDABivariateStatsTest(int n, int m, int li, int lml) {
		this.n = n; 
		this.m = m; 
		this.label_index = li;
		this.label_measurement_level = lml;
	}
	
	@Parameters
	 public static Collection<Object[]> data() {
	   Object[][] data = new Object[][] { { 10000, 100, 1, 1 }, { 10000, 100, 100, 0}, 
			                              { 100000, 100, 1, 1 }, { 100000, 100, 100, 0}
			   							  };
	   return Arrays.asList(data);
	 }
	 
	@Override
	public void setUp() {
		addTestConfiguration(TEST_CLASS_DIR, TEST_NAME);
	}
	
	@Test
	public void testMDABivariateStats() {
		System.out.println("------------ BEGIN " + TEST_NAME + " TEST WITH {" + n + ", " + m
				+ ", " + label_index + ", " + label_measurement_level + "} ------------");
		
		getAndLoadTestConfiguration(TEST_NAME);
		
		List<String> proArgs = new ArrayList<String>();
		proArgs.add("-stats");
		proArgs.add("-args");
		proArgs.add(input("X"));
		proArgs.add(Integer.toString(label_index));
		proArgs.add(input("feature_indices"));
		proArgs.add(Integer.toString(label_measurement_level));
		proArgs.add(input("feature_measurement_levels"));
		proArgs.add(output("stats"));
		proArgs.add(output("tests"));
		proArgs.add(output("covariances"));
		proArgs.add(output("standard_deviations"));
		proArgs.add(output("contingency_tables_counts"));
		proArgs.add(output("contingency_tables_label_values"));
		proArgs.add(output("contingency_tables_feature_values"));
		proArgs.add(output("feature_values"));
		proArgs.add(output("feature_counts"));
		proArgs.add(output("feature_means"));
		proArgs.add(output("feature_standard_deviations"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
        
		fullDMLScriptName = getScript();
		
		rCmd = getRCmd(inputDir(), Integer.toString(label_index), Integer.toString(label_measurement_level), expectedDir());

		double[][] X = getRandomMatrix(n, m, 0, 1, 1, System.currentTimeMillis());
		for(int i=0; i<X.length; i++)
			for(int j=m/2; j<X[i].length; j++){
				//generating a 5-valued categorical random variable
				if(X[i][j] < 0.2) X[i][j] = 1;
				else if(X[i][j] < 0.4) X[i][j] = 2; 
				else if(X[i][j] < 0.6) X[i][j] = 3; 
				else if(X[i][j] < 0.8) X[i][j] = 4;
				else X[i][j] = 5;	
			}
		
		double[][] feature_indices = new double[m-1][1];
		double[][] feature_measurement_levels = new double[m-1][1];
		int pos = 0;
		for(int i=1; i<=m; i++)
			if(i != label_index){
				feature_indices[pos][0] = i;
				feature_measurement_levels[pos][0] = (i > m/2) ? 0 : 1;
				pos++;
			}
		
		MatrixCharacteristics mcX = new MatrixCharacteristics(n, m, -1, -1);
		writeInputMatrixWithMTD("X", X, true, mcX);
		
		MatrixCharacteristics mc_features = new MatrixCharacteristics(m-1, 1, -1, -1);
		writeInputMatrixWithMTD("feature_indices", feature_indices, true, mc_features);
		writeInputMatrixWithMTD("feature_measurement_levels", feature_measurement_levels, true, mc_features);
		
		int expectedNumberOfJobs = -1;
		runTest(true, EXCEPTION_NOT_EXPECTED, null, expectedNumberOfJobs); 
		
		runRScript(true);

		HashMap<CellIndex, Double> statsSYSTEMDS = readDMLMatrixFromHDFS("stats");
		HashMap<CellIndex, Double> statsR = readRMatrixFromFS("stats");
		
		TestUtils.compareMatrices(statsSYSTEMDS, statsR, 0.000001, "statsSYSTEMDS", "statsR");
	}
}
