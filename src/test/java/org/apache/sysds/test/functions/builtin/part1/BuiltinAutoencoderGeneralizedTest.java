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

import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Assert;
import org.junit.Test;

public class BuiltinAutoencoderGeneralizedTest extends AutomatedTestBase {
    private static final String TEST_NAME = "autoencoder_generalized";
    private static final String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinAutoencoderGeneralizedTest.class.getSimpleName() + "/";

    private static final int ROWS = 128;
    private static final int COLS = 64;
    private static final double DENSE = 0.9;

    private static final int EPOCHS = 2;
    private static final int BATCH = 32;
    private static final double STEP = 1e-4;
    private static final double DECAY = 0.95;
    private static final double MOMENTUM = 0.9;

    private static final String MODE_LOCAL = "LOCAL";
    private static final String UTYPE_BSP = "BSP";
    private static final String FREQ_BATCH = "BATCH";
    private static final String SCHEME_RANDOM = "DISJOINT_RANDOM";

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,
                new String[] { "W1_out", "Wlast_out", "hidden_out" }));
    }

    @Test
    public void testDefaultSingleLayer() {
        runAutoencoderTest(new int[] { 16 }, "DEFAULTSERVER");
    }

    @Test
    public void testDefaultTwoLayer() {
        runAutoencoderTest(new int[] { 16, 8 }, "DEFAULTSERVER");
    }

    @Test
    public void testDefaultThreeLayer() {
        runAutoencoderTest(new int[] { 24, 12, 6 }, "DEFAULTSERVER");
    }

    @Test
    public void testParamservSingleLayer() {
        runAutoencoderTest(new int[] { 16 }, "PARAMSERVER");
    }

    @Test
    public void testParamservTwoLayer() {
        runAutoencoderTest(new int[] { 16, 8 }, "PARAMSERVER");
    }

    @Test
    public void testParamservThreeLayer() {
        runAutoencoderTest(new int[] { 24, 12, 6 }, "PARAMSERVER");
    }

    private void runAutoencoderTest(int[] hidden, String method) {
        loadTestConfiguration(getTestConfiguration(TEST_NAME));

        String HOME = SCRIPT_DIR + TEST_DIR;
        fullDMLScriptName = HOME + TEST_NAME + ".dml";

        int h1 = hidden[0];
        int h2 = hidden.length > 1 ? hidden[1] : 0;
        int h3 = hidden.length > 2 ? hidden[2] : 0;
        programArgs = new String[] { "-nvargs",
                "X=" + input("X"),
                "H1=" + h1, "H2=" + h2, "H3=" + h3,
                "EPOCH=" + EPOCHS, "BATCH=" + BATCH, "STEP=" + STEP,
                "DECAY=" + DECAY, "MOMENTUM=" + MOMENTUM,
                "METHOD=" + method, "MODE=" + MODE_LOCAL, "UTYPE=" + UTYPE_BSP,
                "FREQ=" + FREQ_BATCH, "WORKERS=" + 2, "SCHEME=" + SCHEME_RANDOM,
                "NBATCHES=" + 0, "MODELAVG=" + false,
                "W1_out=" + output("W1_out"),
                "Wlast_out=" + output("Wlast_out"),
                "hidden_out=" + output("hidden_out")
        };

        double[][] X = getRandomMatrix(ROWS, COLS, 0, 1, DENSE, 42);
        writeInputMatrixWithMTD("X", X, true);

        runTest(true, false, null, -1);

        MatrixCharacteristics w1Meta = readDMLMetaDataFile("W1_out");
        MatrixCharacteristics wlastMeta = readDMLMetaDataFile("Wlast_out");
        MatrixCharacteristics hiddenMeta = readDMLMetaDataFile("hidden_out");

        Assert.assertEquals(hidden[0], w1Meta.getRows());
        Assert.assertEquals(COLS, w1Meta.getCols());
        Assert.assertEquals(COLS, wlastMeta.getRows());
        Assert.assertEquals(hidden[0], wlastMeta.getCols());
        Assert.assertEquals(ROWS, hiddenMeta.getRows());
        Assert.assertEquals(hidden[hidden.length - 1], hiddenMeta.getCols());
    }
}