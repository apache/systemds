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

package org.apache.sysds.test.functions.frame;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;


@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FrameColNamesPropagationTest extends AutomatedTestBase {
    private final static String TEST_NAME_CBIND = "ColNameCbindPropagation";
    private final static String TEST_NAME_RBIND = "ColNameRbindPropagation";
    private final static String TEST_NAME_SLICE = "ColNameSlicePropagation";
    private final static String TEST_DIR = "functions/frame/";
    private static final String TEST_CLASS_DIR = TEST_DIR + FrameColNamesPropagationTest.class.getSimpleName() + "/";

    @Parameterized.Parameter
    public int _matrixDim;

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
                {10},
                {100},
                {1000},
        });
    }

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME_CBIND, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_CBIND, new String[] {"B"}));
        addTestConfiguration(TEST_NAME_RBIND, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_RBIND, new String[] {"B"}));
        addTestConfiguration(TEST_NAME_SLICE, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_SLICE, new String[] {"B"}));

    }

    @Test
    public void testPropagationCbindCP() {
        runPropagationCbindTest(_matrixDim, ExecType.CP);
    }

    @Test
    public void testPropagationRbindCP() {
        runPropagationRbindTest(_matrixDim, ExecType.CP);
    }

    @Test
    public void testPropagationSliceCP() {
        runPropagationSliceTest(_matrixDim, ExecType.CP);
    }


    private String[] genColnames(int n, String prefix){
        String[] colName = new String[n];
        for(int i = 0; i < n; i++){
            colName[i] = prefix + i;
        }
        return colName;
    }

    private void runPropagationCbindTest(Integer matrixDim, ExecType et) {
        Types.ExecMode platformOld = setExecMode(et);
        boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
        setOutputBuffering(true);
        try {

            // generate an array of column names depending on the dimension of the frame block
            String[] colNames1 = genColnames(matrixDim, "A");
            String[] colNames2 = genColnames(matrixDim, "B");

            getAndLoadTestConfiguration(TEST_NAME_CBIND);
            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME_CBIND + ".dml";


            programArgs = new String[] {"-args",
                    input("X1"), String.valueOf(matrixDim),
                    String.valueOf(matrixDim),
                    input("X2"),
                    Integer.toString(matrixDim),
                    output("B")};

            FrameWriter writer = FrameWriterFactory.createFrameWriter(FileFormat.CSV,
                    new FileFormatPropertiesCSV(true, ",", false));


            Types.ValueType[] schema1 = Collections.nCopies(
                    matrixDim, Types.ValueType.FP64).toArray(new Types.ValueType[0]);
            FrameBlock X1 = new FrameBlock(schema1);
            X1.setColumnNames(colNames1);
            double[][] data_X = getRandomMatrix(matrixDim, matrixDim, Double.MIN_VALUE, Double.MAX_VALUE, 0.7, 14123);
            TestUtils.initFrameData(X1, data_X, schema1, matrixDim);
            writer.writeFrameToHDFS(X1, input("X1"), matrixDim, matrixDim);


            Types.ValueType[] schema2 = Collections.nCopies(
                    matrixDim, Types.ValueType.FP64).toArray(new Types.ValueType[0]);
            FrameBlock X2 = new FrameBlock(schema2);
            X2.setColumnNames(colNames2);
            double[][] data_X2 = getRandomMatrix(matrixDim, matrixDim, Double.MIN_VALUE, Double.MAX_VALUE, 0.7, 14123);
            TestUtils.initFrameData(X2, data_X2, schema2, matrixDim);
            writer.writeFrameToHDFS(X2, input("X2"), matrixDim, matrixDim);


            runTest(true, false, null, -1);


            FrameBlock out = readDMLFrameFromHDFS("B", FileFormat.BINARY);

            // create array of expected column names
            String[] expected = new String[colNames1.length + colNames2.length];
            System.arraycopy(colNames1, 0, expected, 0, colNames1.length);
            System.arraycopy(colNames2, 0, expected, colNames1.length, colNames2.length);

            // compare column names after operation with expected column names
            for(int i = 0; i < expected.length; i++) {
                Assert.assertEquals(
                        "Wrong colName at pos:" + i,
                        expected[i],
                        out.get(0, i).toString()
                );
            }

        }
        catch(Exception ex) {
            throw new RuntimeException(ex);
        }
        finally {
            rtplatform = platformOld;
            DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
        }
    }

    private void runPropagationRbindTest(Integer matrixDim, ExecType et) {
        Types.ExecMode platformOld = setExecMode(et);
        boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
        setOutputBuffering(true);
        try {

            // generate an array of column names depending on the dimension of the frame block
            String[] colNames1 = genColnames(matrixDim, "A");
            String[] colNames2 = genColnames(matrixDim, "B");

            getAndLoadTestConfiguration(TEST_NAME_RBIND);
            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME_RBIND + ".dml";


            programArgs = new String[] {"-args",
                    input("X1"), String.valueOf(matrixDim),
                    String.valueOf(matrixDim),
                    input("X2"),
                    Integer.toString(matrixDim),
                    output("B")};

            FrameWriter writer = FrameWriterFactory.createFrameWriter(FileFormat.CSV,
                    new FileFormatPropertiesCSV(true, ",", false));


            Types.ValueType[] schema1 = Collections.nCopies(
                    matrixDim, Types.ValueType.FP64).toArray(new Types.ValueType[0]);
            FrameBlock X1 = new FrameBlock(schema1);
            X1.setColumnNames(colNames1);
            double[][] data_X = getRandomMatrix(matrixDim, matrixDim, Double.MIN_VALUE, Double.MAX_VALUE, 0.7, 14123);
            TestUtils.initFrameData(X1, data_X, schema1, matrixDim);
            writer.writeFrameToHDFS(X1, input("X1"), matrixDim, matrixDim);


            Types.ValueType[] schema2 = Collections.nCopies(
                    matrixDim, Types.ValueType.FP64).toArray(new Types.ValueType[0]);
            FrameBlock X2 = new FrameBlock(schema2);
            X2.setColumnNames(colNames2);
            double[][] data_X2 = getRandomMatrix(matrixDim, matrixDim, Double.MIN_VALUE, Double.MAX_VALUE, 0.7, 14123);
            TestUtils.initFrameData(X2, data_X2, schema2, matrixDim);
            writer.writeFrameToHDFS(X2, input("X2"), matrixDim, matrixDim);

            runTest(true, false, null, -1);

            FrameBlock out = readDMLFrameFromHDFS("B", FileFormat.BINARY);

            // expected are the column names from the first frame block
            for(int i = 0; i < colNames1.length; i++) {
                Assert.assertEquals(
                        "Wrong colName at pos:" + i,
                        colNames1[i],
                        out.get(0, i).toString()
                );
            }

        }
        catch(Exception ex) {
            throw new RuntimeException(ex);
        }
        finally {
            rtplatform = platformOld;
            DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
        }
    }

    private void runPropagationSliceTest(Integer matrixDim, ExecType et) {
        Types.ExecMode platformOld = setExecMode(et);
        boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
        setOutputBuffering(true);
        try {

            // generate an array of column names depending on the dimension of the frame block
            String[] colNames = genColnames(matrixDim, "A");

            getAndLoadTestConfiguration(TEST_NAME_SLICE);
            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME_SLICE + ".dml";


            programArgs = new String[] {"-args",
                    input("X"), String.valueOf(matrixDim),
                    String.valueOf(matrixDim),
                    output("B")};

            FrameWriter writer = FrameWriterFactory.createFrameWriter(FileFormat.CSV,
                    new FileFormatPropertiesCSV(true, ",", false));


            Types.ValueType[] schema = Collections.nCopies(
                    matrixDim, Types.ValueType.FP64).toArray(new Types.ValueType[0]);
            FrameBlock X1 = new FrameBlock(schema);
            X1.setColumnNames(colNames);
            double[][] data_X = getRandomMatrix(matrixDim, matrixDim, Double.MIN_VALUE, Double.MAX_VALUE, 0.7, 14123);
            TestUtils.initFrameData(X1, data_X, schema, matrixDim);
            writer.writeFrameToHDFS(X1, input("X"), matrixDim, matrixDim);

            runTest(true, false, null, -1);

            FrameBlock out = readDMLFrameFromHDFS("B", FileFormat.BINARY);

            String[] expected = Arrays.copyOfRange(colNames, 1, colNames.length-1);

            // expected are the sliced column names
            for(int i = 0; i < expected.length; i++) {
                Assert.assertEquals(
                        "Wrong colName at pos:" + i,
                        expected[i],
                        out.get(0, i).toString()
                );
            }

        }
        catch(Exception ex) {
            throw new RuntimeException(ex);
        }
        finally {
            rtplatform = platformOld;
            DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
        }
    }

}

