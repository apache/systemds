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

package org.apache.sysds.test.functions.io.parquet;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderParquet;
import org.apache.sysds.runtime.io.FrameReaderParquetParallel;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterParquet;
import org.apache.sysds.runtime.io.FrameWriterParquetParallel;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

import java.io.IOException;

/**
 * This test class verifies that a FrameBlock with different data types is correctly written and read from Parquet files. 
 * It tests both sequential and parallel implementations. In these tests a FrameBlock is created, populated with sample 
 * data, written to a Parquet file, and then read back into a new FrameBlock. The test compares the original and read 
 * data to ensure that schema information is preserved and that data conversion is performed correctly.
 */
public class FrameParquetSchemaTest extends AutomatedTestBase {

    private final static String TEST_NAME = "FrameParquetSchemaTest";
    private final static String TEST_DIR = "functions/io/parquet";
    private final static String TEST_CLASS_DIR = TEST_DIR + FrameParquetSchemaTest.class.getSimpleName() + "/";

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"Rout"}));
    }


    /**
     * Test for sequential writer and reader
     * 
     */
    @Test
    public void testParquetWriteReadAllSchemaTypes() {
        String fname = output("Rout");

        // Define a schema with one column per type
        ValueType[] schema = new ValueType[] {
            ValueType.FP64,
            ValueType.FP32,
            ValueType.INT32,
            ValueType.INT64,
            ValueType.BOOLEAN,
            ValueType.STRING
        };

        // Create an empty frame block with the above schema
        FrameBlock fb = new FrameBlock(schema);

        // Populate frame block
        Object[][] rows = new Object[][] {
            { 1.0,    1.1f, 10,  100L,  true,  "A" },
            { 2.0,    2.1f, 20,  200L,  false, "B" },
            { 3.0,    3.1f, 30,  300L,  true,  "C" },
            { 4.0,    4.1f, 40,  400L,  false, "D" },
            { 5.0,    5.1f, 50,  500L,  true,  "E" }
        };

        for (Object[] row : rows) {
            fb.appendRow(row);
        }

        System.out.println(fb);

        int numRows = fb.getNumRows();
        int numCols = fb.getNumColumns();

        // Write the FrameBlock to a Parquet file using the sequential writer
        try {
            FrameWriter writer = new FrameWriterParquet();
            writer.writeFrameToHDFS(fb, fname, numRows, numCols);
        } 
        catch (IOException e) {
            e.printStackTrace();
            Assert.fail("Failed to write frame block to Parquet: " + e.getMessage());
        }

        // Read the Parquet file back into a new FrameBlock
        FrameBlock fbRead = null;
        try {
            FrameReader reader = new FrameReaderParquet();
            String[] colNames = fb.getColumnNames();
            fbRead = reader.readFrameFromHDFS(fname, schema, colNames, numRows, numCols);
        } 
        catch (IOException e) {
            e.printStackTrace();
            Assert.fail("Failed to read frame block from Parquet: " + e.getMessage());
        }

        // Compare the original and the read frame blocks
        compareFrameBlocks(fb, fbRead, 1e-6);
    }

    /**
     * Test for multithreaded writer and reader
     * 
     */
    @Test
    public void testParquetWriteReadAllSchemaTypesParallel() {
        String fname = output("Rout_parallel");

        ValueType[] schema = new ValueType[] {
            ValueType.FP64,
            ValueType.FP32,
            ValueType.INT32,
            ValueType.INT64,
            ValueType.BOOLEAN,
            ValueType.STRING
        };

        FrameBlock fb = new FrameBlock(schema);

        Object[][] rows = new Object[][] {
            { 1.0,    1.1f, 10,  100L,  true,  "A" },
            { 2.0,    2.1f, 20,  200L,  false, "B" },
            { 3.0,    3.1f, 30,  300L,  true,  "C" },
            { 4.0,    4.1f, 40,  400L,  false, "D" },
            { 5.0,    5.1f, 50,  500L,  true,  "E" }
        };

        for (Object[] row : rows) {
            fb.appendRow(row);
        }

        int numRows = fb.getNumRows();
        int numCols = fb.getNumColumns();

        try {
            FrameWriter writer = new FrameWriterParquetParallel();
            writer.writeFrameToHDFS(fb, fname, numRows, numCols);
        } 
        catch (IOException e) {
            e.printStackTrace();
            Assert.fail("Failed to write frame block to Parquet (parallel): " + e.getMessage());
        }
    
        FrameBlock fbRead = null;
        try {
            FrameReader reader = new FrameReaderParquetParallel();
            String[] colNames = fb.getColumnNames();
            fbRead = reader.readFrameFromHDFS(fname, schema, colNames, numRows, numCols);
        } 
        catch (IOException e) {
            e.printStackTrace();
            Assert.fail("Failed to read frame block from Parquet (parallel): " + e.getMessage());
        }

        compareFrameBlocks(fb, fbRead, 1e-6);
    }

    private void compareFrameBlocks(FrameBlock expected, FrameBlock actual, double eps) {
        Assert.assertEquals("Number of rows mismatch", expected.getNumRows(), actual.getNumRows());
        Assert.assertEquals("Number of columns mismatch", expected.getNumColumns(), actual.getNumColumns());

        int rows = expected.getNumRows();
        int cols = expected.getNumColumns();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                Object expVal = expected.get(i, j);
                Object actVal = actual.get(i, j);
                ValueType vt = expected.getSchema()[j];

                // Handle nulls first
                if(expVal == null || actVal == null) {
                    Assert.assertEquals("Mismatch at (" + i + "," + j + ")", expVal, actVal);
                } else {
                    switch(vt) {
                        case FP64:
                        case FP32:
                            double dExp = ((Number) expVal).doubleValue();
                            double dAct = ((Number) actVal).doubleValue();
                            Assert.assertEquals("Mismatch at (" + i + "," + j + ")", dExp, dAct, eps);
                            break;
                        case INT32:
                        case INT64:
                            long lExp = ((Number) expVal).longValue();
                            long lAct = ((Number) actVal).longValue();
                            Assert.assertEquals("Mismatch at (" + i + "," + j + ")", lExp, lAct);
                            break;
                        case BOOLEAN:
                            boolean bExp = (Boolean) expVal;
                            boolean bAct = (Boolean) actVal;
                            Assert.assertEquals("Mismatch at (" + i + "," + j + ")", bExp, bAct);
                            break;
                        case STRING:
                            Assert.assertEquals("Mismatch at (" + i + "," + j + ")", expVal.toString(), actVal.toString());
                            break;
                        default:
                            Assert.fail("Unsupported type in comparison: " + vt);
                    }
                }
            }
        }
    }
}
