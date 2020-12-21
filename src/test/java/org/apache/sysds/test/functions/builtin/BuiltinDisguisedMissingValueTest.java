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

import java.util.HashMap;

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

public class BuiltinDisguisedMissingValueTest extends AutomatedTestBase {

    private final static String TEST_NAME = "disguisedMissingValue";
    private final static String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinOutlierTest.class.getSimpleName() + "/";

    private final static Types.ValueType[] schemaStrings = {Types.ValueType.STRING};
    private final static Types.ValueType[] schemaInteger = {Types.ValueType.INT32};
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
    public void dmvWithStrings() {
        runMissingValueTest( TestType.STRING, ExecType.CP );
    }

    @Test
    public void dmvWithIntegers() {
        runMissingValueTest( TestType.INTEGER, ExecType.CP );
    }

    private void runMissingValueTest(TestType type, ExecType et)
    {
        Types.ExecMode platformOld = setExecMode(et);

        try {
            getAndLoadTestConfiguration(TEST_NAME);

            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[] { "-stats","-args", input("A"), output("O"), output("I")};

            double[][] A = getRandomMatrix(rows, 1, 0, 1, 1, 2);

            switch (type) {
                case STRING:
                    writeInputFrameWithMTD("A", A, true, schemaStrings, Types.FileFormat.CSV);
                    break;
                case INTEGER:
                    writeInputFrameWithMTD("A", A, true, schemaInteger, Types.FileFormat.CSV);
                    break;
            }

            runTest(true, false, null, -1);

            FrameBlock outputFrame = readDMLFrameFromHDFS("O", Types.FileFormat.CSV);
            FrameBlock inputFrame = readDMLFrameFromHDFS("I", Types.FileFormat.CSV);

            String[] output = (String[])outputFrame.getColumnData(0);
            String[] input = (String[])inputFrame.getColumnData(0);

            for(int i = 0; i<input.length; i++)
            {
                TestUtils.compareScalars(null, output[i]);
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
