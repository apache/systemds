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

package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinUnionTest  extends AutomatedTestBase {
    private final static String TEST_NAME = "union";
    private final static String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinUnionTest.class.getSimpleName() + "/";

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"R"}));
    }

    @Test
    public void testUnion1CP() {
        double[][] X = {{1}, {2}, {3}};
        double[][] Y = {{2}, {3}, {4}};
        runUnionTests(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testUnion1SP() {
        double[][] X = {{1}, {2}, {3}};
        double[][] Y = {{2}, {3}, {4}};

        runUnionTests(X, Y, Types.ExecType.SPARK);
    }

    //TODO
    //Fails because R produces different order for union and unique.
    //Therefore do not use unique internally. /:

    @Test
    public void testUnion2CP() {
        double[][] X = {{9}, {2}, {3}};
        double[][] Y = {{2}, {3}, {4}};

        runUnionTests(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testUnion2SP() {
        double[][] X = {{9}, {2}, {3}};
        double[][] Y = {{2}, {3}, {4}};

        runUnionTests(X, Y, Types.ExecType.SPARK);
    }

    @Test
    public void testUnion3CP() {
        double[][] X =  {{12},{22},{13},{4},{6},{7},{8},{9},{12},{12}};
        double[][] Y = {{1},{2},{11},{12},{13},{18},{20},{21},{12}};
        runUnionTests(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testUnion3Spark() {
        double[][] X = {{12},{22},{13},{4},{6},{7},{8},{9},{12},{12}};
        double[][] Y = {{1},{2},{11},{12},{13},{18},{20},{21},{12}};
        runUnionTests(X, Y, Types.ExecType.SPARK);
    }
    
    private void runUnionTests(double[][] X, double[][]Y, Types.ExecType instType) {
        Types.ExecMode platformOld = setExecMode(instType);
        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME));
            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[]{ "-args", input("X"),input("Y"), output("R")};
            fullRScriptName = HOME + TEST_NAME + ".R";
            rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

            writeInputMatrixWithMTD("X", X, true);
            writeInputMatrixWithMTD("Y", Y, true);

            runTest(true, false, null, -1);
            runRScript(true);

            HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
            HashMap<MatrixValue.CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");
            TestUtils.compareMatrices(dmlfile, rfile, 1e-10, "dml", "expected");
        }
        finally {
            rtplatform = platformOld;
        }
    }
}