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
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

public class BuiltinDecisionTreeRealDataTest extends AutomatedTestBase {
	private final static String TEST_NAME = "decisionTreeRealData";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinDecisionTreeRealDataTest.class.getSimpleName() + "/";

	private final static String TITANIC_DATA = DATASET_DIR + "titanic/titanic.csv";
	private final static String TITANIC_TFSPEC = DATASET_DIR + "titanic/tfspec.json";
	private final static String WINE_DATA = DATASET_DIR + "wine/winequality-red-white.csv";
	private final static String WINE_TFSPEC = DATASET_DIR + "wine/tfspec.json";

	
	@Override
	public void setUp() {
		for(int i=1; i<=2; i++)
			addTestConfiguration(TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testDecisionTreeTitanic_MaxV1() {
		runDecisionTree(1, TITANIC_DATA, TITANIC_TFSPEC, 0.875, 1, 1.0, ExecType.CP);
	}
	
	@Test
	public void testRandomForestTitanic1_MaxV1() {
		//one tree with sample_frac=1 should be equivalent to decision tree
		runDecisionTree(1, TITANIC_DATA, TITANIC_TFSPEC, 0.875, 2, 1.0, ExecType.CP);
	}
	
	@Test
	public void testRandomForestTitanic8_MaxV1() {
		//8 trees with sample fraction 0.125 each, accuracy 0.785 due to randomness
		runDecisionTree(1, TITANIC_DATA, TITANIC_TFSPEC, 0.793, 9, 1.0, ExecType.CP);
	}
	
	@Test
	public void testDecisionTreeTitanic_MaxV06() {
		runDecisionTree(1, TITANIC_DATA, TITANIC_TFSPEC, 0.871, 1, 0.6, ExecType.CP);
	}
	
	@Test
	public void testRandomForestTitanic1_MaxV06() {
		//one tree with sample_frac=1 should be equivalent to decision tree
		runDecisionTree(1, TITANIC_DATA, TITANIC_TFSPEC, 0.871, 2, 0.6, ExecType.CP);
	}
	
	@Test
	public void testRandomForestTitanic8_MaxV06() {
		//8 trees with sample fraction 0.125 each, accuracy 0.785 due to randomness
		runDecisionTree(1, TITANIC_DATA, TITANIC_TFSPEC, 0.793, 9, 0.6, ExecType.CP);
	}
	
	@Test
	public void testDecisionTreeWine_MaxV1() {
		runDecisionTree(2, WINE_DATA, WINE_TFSPEC, 0.989, 1, 1.0, ExecType.CP);
	}
	
	@Test
	public void testRandomForestWine_MaxV1() {
		//one tree with sample_frac=1 should be equivalent to decision tree
		runDecisionTree(2, WINE_DATA, WINE_TFSPEC, 0.989, 2, 1.0, ExecType.CP);
	}

	private void runDecisionTree(int test, String data, String tfspec, double minAcc, int dt, double maxV, ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME+test));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + (TEST_NAME+test) + ".dml";
			programArgs = new String[] {"-stats",
				"-args", data, tfspec, String.valueOf(dt), String.valueOf(maxV), output("R")};

			runTest(true, false, null, -1);

			double acc = readDMLMatrixFromOutputDir("R").get(new CellIndex(1,1));
			Assert.assertTrue(acc >= minAcc);
			Assert.assertEquals(0, Statistics.getNoOfExecutedSPInst());
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
