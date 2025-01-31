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

import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.test.TestConfiguration;

import java.util.*;

@RunWith(Parameterized.class)
@net.jcip.annotations.NotThreadSafe

public class BuiltinImageTransformMatrixTest extends AutomatedTestBase {
    private final static String TEST_NAME_LINEARIZED = "image_transform_matrix";
    private final static String TEST_DIR = "functions/builtin/";
    private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinImageTransformMatrixTest.class.getSimpleName() + "/";

    @Parameterized.Parameter(0)
    public double[][] transMat;
    @Parameterized.Parameter(1)
    public double[][] dimMat;
    @Parameterized.Parameter(2)
    public boolean fails;

    private final static double [][] t1 = new double[][] {{2,0,0},{0,1,0},{0,0,1}}; //initial test for warmup
    private final static double [][] d1 = new double[][] {{10, 10},{15,15}};
    private final static double [][] t2 = new double[][] {{2,0,0},{0,1,0},{0,0,1}};
    private final static double [][] d2 = new double[][] {{100, 100},{100,100}}; //test1: 100x100
    private final static double [][] t3 = new double[][] {{2,0,0},{0,1,0},{0,0,1}};
    private final static double [][] d3 = new double[][] {{640, 480},{640,480}}; //test2: 640x480
    private final static double [][] t4 = new double[][] {{4,0,0},{0,1,0},{0,0,1}};
    private final static double [][] d4 = new double[][] {{1280, 720},{1280, 720}};//test3: 1280x720
    private final static double [][] t5 = new double[][] {{2,0,0},{0,1,0},{0,0,1}};
    private final static double [][] d5 = new double[][] {{1920, 1080},{1920, 1080}};//test4 1920x1080
    private final static double [][] t6 = new double[][] {{2,0,0},{0,1,0},{0,0,1}};
    private final static double [][] d6 = new double[][] {{2560, 1440},{2560, 1440}};//test5 2560x1440
    private final static double [][] t7 = new double[][] {{2,0,0},{0,1,0},{0,0,1}};
    private final static double [][] d7 = new double[][] {{3840, 2160},{3840, 2160}}; //test6 3840x2160
    private final static double [][] t8 = new double[][] {{2,0,0},{0,1,0},{0,0,1}};
    private final static double [][] d8 = new double[][] {{3840, 2160},{1980, 1080}};
    private final static double [][] t9 = new double[][] {{2,0,0},{0,1,0},{0,0,1}};
    private final static double [][] d9 = new double[][] {{20.10, 200},{200, 200}};
    private final static double [][] t10 = new double[][] {{0,0,0},{0,0,0},{0,0,0}};
    private final static double [][] d10 = new double[][] {{20.10, 200},{200, 200}};
    private final static double [][] t11 = new double[][] {{0,0,0},{0,1,0},{0,0,0}};
    private final static double [][] d11 = new double[][] {{20.10, 200},{200, 200}};
    private final static double [][] t12 = new double[][] {{2,0,0},{0,1,0},{0,0,1}};
    private final static double [][] d12 = new double[][] {{0, 0},{0, 0}};
    private final static double [][] t13 = new double[][] {{2,0,0},{0,1,0},{0,0,1}};
    private final static double [][] d13 = new double[][] {{0.10, 200},{200, 200}};

    public double internal = 1.0;
    public boolean compareResults = false;

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {{t1, d1, false},{t2, d2, false},{t3, d3, false},{t4, d4, false},
                {t5, d5,false},{t6, d6, false},{t7, d7,false},{t8, d8,false},
                {t9, d9,true},{t10,d10,true},{t11, d11,true},{t12, d12,true},{t13, d13,true}});
    }

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME_LINEARIZED,
                new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_LINEARIZED, new String[] {"B_x"}));
    }

    //test for using the internal implementation only
    @Test
    public void testImageTransformMatrix() {
        runImageTransformMatrixTest(ExecType.CP);
    }

    //test for using the script implementation only
    @Test
    public void testImageTransformMatrixScript() {
        internal = 0;
        runImageTransformMatrixTest(ExecType.CP);
    }

    //test for comparing the script and internal implementations for correctness
    //presumably due to caching it should not be used for benchmarks
    @Test
    public void testImageTransformMatrixCompare() {
        internal = 1;
        compareResults = true;
        runImageTransformMatrixTest(ExecType.CP);
    }

    private void runImageTransformMatrixTest(ExecType instType) {
        ExecMode platformOld = setExecMode(instType);
        disableOutAndExpectedDeletion();

        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME_LINEARIZED));

            String HOME = SCRIPT_DIR + TEST_DIR;

            writeInputMatrixWithMTD("transMat", transMat, true);
            writeInputMatrixWithMTD("dimMat", dimMat, true);
            writeInputMatrixWithMTD("internal", new double[][] {{internal}}, true);

            fullDMLScriptName = HOME + TEST_NAME_LINEARIZED + ".dml";
            programArgs = new String[]{"-nvargs", "transMat=" + input("transMat"), "dimMat=" + input("dimMat"), "out_file=" + output("B_internal"),"internal=" + input("internal"), "--debug"};
            runTest(true, fails, null, -1);
            if (compareResults && !fails) {
                internal = 0;
                writeInputMatrixWithMTD("internal", new double[][] {{internal}}, true);
                programArgs = new String[]{"-nvargs", "transMat=" + input("transMat"), "dimMat=" + input("dimMat"), "out_file=" + output("B_script"),"internal=" + input("internal"), "--debug"};
                runTest(true, fails, null, -1);

                HashMap<MatrixValue.CellIndex, Double> internalfile = readDMLMatrixFromOutputDir("B_internal");
                HashMap<MatrixValue.CellIndex, Double> scriptfile = readDMLMatrixFromOutputDir("B_script");
                TestUtils.compareMatrices(internalfile, scriptfile, 1e-10, "Stat-DML", "Stat-R");
            }

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            rtplatform = platformOld;
            internal = 1.0;
        }
    }


}