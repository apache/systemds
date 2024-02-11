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

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;




public class BuiltinHmmInnitTest extends AutomatedTestBase {

    private final static String TEST_NAME = "HmmInnit";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinHmmInnitTest.class.getSimpleName() + "/";

    private final static double eps = 1e-3;
    private final static String observation_count = "10";
    private final static String hiddenstates_count = "10";

    @Override
	public void setUp() {
        addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR , TEST_NAME , new String[]{"B"}));
	}

	@Test
	public void testVerboseGridSearchLmCodegenCP() {
		runHmmInnit(TEST_NAME, ExecMode.SINGLE_NODE, true, false);
	}

    @Test
	public void testVerboseGridSearchLmCP() {
		runHmmInnit(TEST_NAME, ExecMode.SINGLE_NODE, true, false);
	}
	
	@Test
	public void testVerboseGridSearchLmHybrid() {
		runHmmInnit(TEST_NAME, ExecMode.HYBRID, true, false);
	}
    private void runHmmInnit(String testname, ExecMode execMode, boolean verbose, boolean random_mode) {
        ExecMode modeOld = setExecMode(execMode);

        try{
            loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;

            fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", "observation_count=" + observation_count, 
                                       "hiddenstates_count=" + hiddenstates_count, "verbose=" + verbose, 
                                       "random_mode=" + random_mode, "sp_path=" + output("start_prob"),
                                       "tp_path=" + output("transition_prob"), "ep_path=" + output("emission_prob")};

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(observation_count, hiddenstates_count, expectedDir());

            runTest(true, false, null, -1);

            HashMap<CellIndex, Double> dml_start_prob = readDMLMatrixFromOutputDir("start_prob");
            HashMap<CellIndex, Double> dml_transition_prob = readDMLMatrixFromOutputDir("transition_prob");
            HashMap<CellIndex, Double> dml_emission_prob = readDMLMatrixFromOutputDir("emission_prob");

            if(!random_mode){
                runRScript(true);
                HashMap<CellIndex, Double> r_start_prob = readRMatrixFromExpectedDir("start_prob");
                HashMap<CellIndex, Double> r_transition_prob = readRMatrixFromExpectedDir("transition_prob");
                HashMap<CellIndex, Double> r_emission_prob = readRMatrixFromExpectedDir("emission_prob");

                TestUtils.compareMatrices(dml_start_prob, r_start_prob, eps, "ml_start_prob", "r_start_prob");
                TestUtils.compareMatrices(dml_transition_prob, r_transition_prob, eps, "dml_transition_prob", "r_transition_prob");
                TestUtils.compareMatrices(dml_emission_prob, r_emission_prob, eps, "dml_emission_prob", "r_emission_prob");
            } 

            if(verbose) 
				Assert.assertTrue(Statistics.getCPHeavyHitterCount("print")>5);

        }
        finally {
            resetExecMode(modeOld);
        }
    }
 }