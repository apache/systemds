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

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class BuiltinHmmFitTest extends AutomatedTestBase {
    
    private final static String TEST_NAME = "hmmFit";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinHmmFitTest.class.getSimpleName() + "/";

	private final static Integer observation_length = 20;    
	private final static Integer observation_count = 10;
    private final static Integer hiddenstates_count = 10;
    private final static Integer num_iterations = 10;
	private final static double eps = 1e-3;

    @Override
	public void setUp() {
        addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR , TEST_NAME , new String[]{"B"}));
	}

    @Test
	public void testVerboseHmmFitCP() {
		runHmmFit(TEST_NAME, ExecMode.SINGLE_NODE, true);
	}
	
	@Test
	public void testVerboseHmmFitHybrid() {
		runHmmFit(TEST_NAME, ExecMode.HYBRID, true);
	}


    private void runHmmFit(String testname, ExecMode execMode, boolean verbose) {

		ExecMode modeOld = setExecMode(execMode);

        try{
            loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			fullRScriptName = HOME + TEST_NAME + ".R";

			double[][] observation_sequence = getRandomMatrix(1, observation_length, 1, observation_count, 1, -1);
			double[][] start_prob = new double[1][hiddenstates_count];
			double[][] transition_prob = new double[hiddenstates_count][hiddenstates_count];
			double[][] emission_prob = new double[hiddenstates_count][observation_count];

			Arrays.fill(start_prob[0], 1.0 / hiddenstates_count);
			for (int i = 0; i < hiddenstates_count; i++) {
				for (int j = 0; j < hiddenstates_count; j++) {
					transition_prob[i][j] = 0.7 / hiddenstates_count;
					if (i == j) { 
						transition_prob[i][j] += 0.3;
					}
				}
				for (int j = 0; j < observation_count; j++) {
					emission_prob[i][j] = 1.0 / observation_count;
				}
			}

			writeInputMatrixWithMTD("observation_sequence", observation_sequence, true);
			writeInputMatrixWithMTD("start_prob", start_prob, true);
			writeInputMatrixWithMTD("transition_prob", transition_prob, true);
			writeInputMatrixWithMTD("emission_prob", emission_prob, true);

			List<String> proArgs = new ArrayList<>();
			proArgs.add("-args");
			proArgs.add("os_input_path=" + input("observation_sequence"));
			proArgs.add("sp_input_path=" + input("start_prob"));
			proArgs.add("tp_input_path=" + input("transition_prob"));
			proArgs.add("ep_input_path=" + input("emission_prob"));
			proArgs.add("num_iterations=" + String.valueOf(num_iterations));
			proArgs.add("verbose=" + String.valueOf(verbose));
			proArgs.add("ep_output_path=" + output("emission_prob"));
			proArgs.add("tp_output_path=" + output("transition_prob"));
			proArgs.add("diff_output_path=" + output("diff_array"));
            programArgs = proArgs.toArray(new String[proArgs.size()]);

			rCmd = getRCmd(inputDir(), String.valueOf(num_iterations), String.valueOf(observation_count), String.valueOf(hiddenstates_count), expectedDir());

			runTest(true, false, null, -1);

			HashMap<CellIndex, Double> dml_emission_prob = readDMLMatrixFromOutputDir("emission_prob");
            HashMap<CellIndex, Double> dml_transition_prob = readDMLMatrixFromOutputDir("transition_prob");
            HashMap<CellIndex, Double> dml_diff_array = readDMLMatrixFromOutputDir("diff_array");

			HashMap<CellIndex, Double> r_emission_prob = readRMatrixFromExpectedDir("emission_prob");
			HashMap<CellIndex, Double> r_transition_prob = readRMatrixFromExpectedDir("transition_prob");
			HashMap<CellIndex, Double> r_diff_array = readRMatrixFromExpectedDir("diff_array");

			TestUtils.compareMatrices(dml_diff_array, r_diff_array, eps, "ml_start_prob", "r_start_prob");
			TestUtils.compareMatrices(dml_transition_prob, r_transition_prob, eps, "dml_transition_prob", "r_transition_prob");
			TestUtils.compareMatrices(dml_emission_prob, r_emission_prob, eps, "dml_emission_prob", "r_emission_prob");

			if(verbose) {
				Assert.assertTrue(Statistics.getCPHeavyHitterCount("print")>5);
            }
		}
        finally {
            resetExecMode(modeOld);
        }
    }
}
