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

package org.apache.sysds.test.functions.lineage;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.junit.Test;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class UnmarkLoopDepVarsTest extends AutomatedTestBase {
	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME1 = "unmarkLoopDepVars";
	
	protected String TEST_CLASS_DIR = TEST_DIR + UnmarkLoopDepVarsTest.class.getSimpleName() + "/";
	
	protected static final int numRecords = 100;
	protected static final int numFeatures = 30;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1));
	}
	
	@Test
	public void unmarkloopdepvars() {
		runtest();
	}

	private void runtest() {
		try {
			getAndLoadTestConfiguration(UnmarkLoopDepVarsTest.TEST_NAME1);
			List<String> proArgs = new ArrayList<>();
			
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add("-args");
			proArgs.add(input("X"));
			proArgs.add(output("Res"));
			programArgs = proArgs.toArray(new String[0]);
			fullDMLScriptName = getScript();
			double[][] X = getRandomMatrix(numRecords, numFeatures, 0, 1, 0.8, -1);
			writeInputMatrixWithMTD("X", X, true);
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			HashMap<MatrixValue.CellIndex, Double> R_orig = readDMLMatrixFromHDFS("Res");

			proArgs.clear();
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add(ReuseCacheType.REUSE_FULL.name().toLowerCase());
			proArgs.add("-args");
			proArgs.add(input("X"));
			proArgs.add(output("Res"));
			programArgs = proArgs.toArray(new String[0]);
			fullDMLScriptName = getScript();
			writeInputMatrixWithMTD("X", X, true);
			Lineage.resetInternalState();
			Lineage.setLinReusePartial();
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			Lineage.setLinReuseNone();
			HashMap<MatrixValue.CellIndex, Double> R_reused = readDMLMatrixFromHDFS("Res");
			TestUtils.compareMatrices(R_orig, R_reused, 1e-6, "Origin", "Reused");
		}
		finally {
			Recompiler.reinitRecompiler();
		}
	}
}
