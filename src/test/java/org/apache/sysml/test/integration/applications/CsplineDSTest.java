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

package org.apache.sysml.test.integration.applications;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

import org.junit.runners.Parameterized.Parameters;

import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;

public abstract class CsplineDSTest  extends AutomatedTestBase {

    protected final static String TEST_DIR = "applications/cspline/";
    protected final static String TEST_NAME = "CsplineDS";
    protected String TEST_CLASS_DIR = TEST_DIR + CsplineDSTest.class.getSimpleName() + "/";

    protected int numRecords, numDim;

    public CsplineDSTest(int rows, int cols) {
        numRecords = rows;
        numDim = 1; // we have cubic spline which is always one dimensional
    }

    @Parameters
    public static Collection<Object[]> data() {
        Object[][] data = new Object[][] {
                {10, 1},
                {100, 1},
                {1000, 1},
        };
        return Arrays.asList(data);
    }

    @Override
    public void setUp() {
        addTestConfiguration(TEST_CLASS_DIR, TEST_NAME);
    }

    protected void testCsplineDS(ScriptType scriptType)
    {
		System.out.println("------------ BEGIN " + TEST_NAME + " " + scriptType + " TEST WITH {" + numRecords + ", " + numDim
				+ "} ------------");
		this.scriptType = scriptType;

        int rows = numRecords;
        int cols = numDim;

        getAndLoadTestConfiguration(TEST_NAME);

		List<String> proArgs = new ArrayList<String>();
		if (scriptType == ScriptType.PYDML) {
			proArgs.add("-python");
		}
		proArgs.add("-nvargs");
		proArgs.add("X=" + input("X"));
		proArgs.add("Y=" + input("Y"));
		proArgs.add("K=" + output("K"));
		proArgs.add("O=" + output("pred_y"));
		proArgs.add("inp_x=" + 4.5);
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		
		fullDMLScriptName = getScript();
		
		rCmd = getRCmd(input("X.mtx"), input("Y.mtx"), Double.toString(4.5), expected("pred_y"));

        double[][] X = new double[rows][cols];

        // X axis is given in the increasing order
        for (int rid = 0; rid < rows; rid++) {
            for (int cid = 0; cid < cols; cid++) {
                X[rid][cid] = rid+1;
            }
        }

        double[][] Y = getRandomMatrix(rows, cols, 0, 5, 1.0, -1);

        writeInputMatrixWithMTD("X", X, true);
        writeInputMatrixWithMTD("Y", Y, true);

        runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

        runRScript(true);

        HashMap<CellIndex, Double> priorR = readRMatrixFromFS("pred_y");
        HashMap<CellIndex, Double> priorSYSTEMML= readDMLMatrixFromHDFS("pred_y");

        TestUtils.compareMatrices(priorR, priorSYSTEMML, Math.pow(10, -12), "k_R", "k_SYSTEMML");
    }
}
