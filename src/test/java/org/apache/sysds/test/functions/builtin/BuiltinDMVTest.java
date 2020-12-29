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

import java.sql.Array;
import java.util.Arrays;
import java.util.HashMap;

import org.apache.hadoop.yarn.webapp.hamlet.Hamlet;
import org.apache.sysds.api.mlcontext.Frame;
import org.apache.sysds.api.mlcontext.Matrix;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.functions.binary.frame.FrameMapTest;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class BuiltinDMVTest extends AutomatedTestBase {

    private final static String TEST_NAME = "disguisedMissingValue";
    private final static String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinOutlierTest.class.getSimpleName() + "/";

    private final static Types.ValueType[] schemaStrings = {Types.ValueType.STRING};
    private final static int rows = 10;

    static enum TestType {
        STRING,
        INTEGER
    }

    @BeforeClass
    public static void init() {
        TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
    }

    @AfterClass
    public static void cleanUp() {
        if (TEST_CACHE_ENABLED) {
            TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
        }
    }

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
        if (TEST_CACHE_ENABLED) {
            setOutAndExpectedDeletionDisabled(true);
        }
    }

    @Test
    public void IntegerFrameTest() {
        String[] content = new String[]{"44","3","235","52","weg","12", "11", "33", "22", "99"};

        FrameBlock f = new FrameBlock(schemaStrings);
        for (String s : content) {
            f.appendRow(new String[]{s});
        }
        System.out.println(f.getColumnData(0));

        int[][] positions = new int[1][1];
        positions[0] = new int[]{4};
        runMissingValueTest(f, ExecType.CP, positions);
    }

    @Test
    public void AdvancedIntegerFrameTest() {
        String[] content1 = new String[]{"44","3","235","52","weg","12", "11", "33", "22", "99"};

        FrameBlock f = new FrameBlock(schemaStrings);
        for (String s : content1) {
            f.appendRow(new String[]{s});
        }
        // f.appendColumn(new String[]{"15","weeeg","111","52","weg","333", "11", "999", "22", "99"});

        int[][] positions = new int [2][2];
        positions[0] = new int[]{4};
        //positions[1] = new int[]{1,5};

        //hardcoded for now
        runMissingValueTest(f, ExecType.CP, positions);
    }

    private void runMissingValueTest(FrameBlock test_frame, ExecType et, int[][] positions)
    {
        Types.ExecMode platformOld = setExecMode(et);

        try {
            getAndLoadTestConfiguration(TEST_NAME);

            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[] { "-stats","-args", input("A"), output("O")};

            FrameWriterFactory.createFrameWriter(Types.FileFormat.CSV).
                    writeFrameToHDFS(test_frame, input("A"), test_frame.getNumRows(), test_frame.getNumColumns());

            runTest(true, false, null, -1);

            FrameBlock outputFrame = readDMLFrameFromHDFS("O", Types.FileFormat.CSV);

            String[] output = (String[])outputFrame.getColumnData(0);

            for (int[] position : positions) {
                for (int i = 0; i < position.length; i++)
                {
                    // if it's NA then it will be null here - otherwise the value you chose for disguised..
                    TestUtils.compareScalars(null, output[position[i]]);
                }
            }

        }
        catch (Exception ex) {
            throw new RuntimeException(ex);
        }
        finally {
            resetExecMode(platformOld);
        }
    }

}
