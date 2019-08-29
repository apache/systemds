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
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestUtils;

@RunWith(value = Parameterized.class)
public class CsplineCGTest extends AutomatedTestBase {
    protected final static String TEST_DIR = "applications/cspline/";
    protected final static String TEST_NAME = "CsplineCG";
    protected String TEST_CLASS_DIR = TEST_DIR + CsplineCGTest.class.getSimpleName() + "/";
    protected int numRecords, numDim;
    public CsplineCGTest(int rows, int cols) {
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

    @Test
    public void testCsplineCG()
    {
		System.out.println("------------ BEGIN " + TEST_NAME + " TEST WITH {" + numRecords + ", " + numDim
				+ "} ------------");
		
        int rows = numRecords;
        int cols = numDim;
        int numIter = rows; // since CG will converse in worse case n

        getAndLoadTestConfiguration(TEST_NAME);
        
		List<String> proArgs = new ArrayList<String>();
		proArgs.add("-nvargs");
		proArgs.add("X=" + input("X"));
		proArgs.add("Y=" + input("Y"));
		proArgs.add("K=" + output("K"));
		proArgs.add("O=" + output("pred_y"));
		proArgs.add("maxi=" + numIter);
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

        HashMap<CellIndex, Double> pred_y_R = readRMatrixFromFS("pred_y");
        HashMap<CellIndex, Double> pred_y_SYSTEMDS= readDMLMatrixFromHDFS("pred_y");

        TestUtils.compareMatrices(pred_y_R, pred_y_SYSTEMDS, Math.pow(10, -5), "k_R", "k_SYSTEMDS");
    }
}
