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

package org.apache.sysds.test.functions.updateinplace;

import java.util.HashMap;

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;


public class UnaryUpdateInPlaceTest extends AutomatedTestBase{
	private final static String TEST_NAME = "UnaryUpdateInplace";
	private final static String TEST_DIR = "functions/updateinplace/";
	private final static String TEST_CLASS_DIR = TEST_DIR + UnaryUpdateInPlaceTest.class.getSimpleName() + "/";
	private final static double eps = 1e-3;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B",}));
	}

	@Test
	public void testInPlace() {
		runInPlaceTest(Types.ExecType.CP);
	}


	private void runInPlaceTest(Types.ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);
		boolean oldFlag = OptimizerUtils.ALLOW_UNARY_UPDATE_IN_PLACE;
		
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-nvargs","Out=" + output("Out") };

			OptimizerUtils.ALLOW_UNARY_UPDATE_IN_PLACE = true;
			runTest(true, false, null, -1);
			HashMap<MatrixValue.CellIndex, Double> dmlfileOut1 = readDMLMatrixFromOutputDir("Out");
			OptimizerUtils.ALLOW_UNARY_UPDATE_IN_PLACE = false;
			runTest(true, false, null, -1);
			HashMap<MatrixValue.CellIndex, Double> dmlfileOut2 = readDMLMatrixFromOutputDir("Out");

			//compare matrices
			TestUtils.compareMatrices(dmlfileOut1,dmlfileOut2,eps,"Stat-DML1","Stat-DML2");
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		finally {
			rtplatform = platformOld;
			OptimizerUtils.ALLOW_UNARY_UPDATE_IN_PLACE = oldFlag;
		}
	}
}
