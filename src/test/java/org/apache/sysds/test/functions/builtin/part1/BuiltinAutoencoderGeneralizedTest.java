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
    private static final String TEST_NAME = "autoencoderGeneralized";
    private static final String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinAutoencoderGeneralizedTest.class.getSimpleName() + "/";

    private static final int ROWS = 128;
    private static final int COLS = 64;
    private static final double DENSE = 0.9;

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,
                new String[] { "W1_out", "Wlast_out", "hidden_out" }));
    }

    @Test
    public void testAutoencoderThreeLayerOutputs() {
        runAutoencoderTest("DEFAULTSERVER", 16, 8, 4, 2, DENSE);
    }

    @Test
    public void testAutoencoderTwoLayerOutputs() {
        runAutoencoderTest("DEFAULTSERVER", 16, 8, 0, 0, DENSE);
    }

    @Test
    public void testAutoencoderSingleLayerOutputs() {
        runAutoencoderTest("DEFAULTSERVER", 16, 0, 0, 1, DENSE);
    }

    @Test
    public void testAutoencoderSparseInputOutputs() {
        runAutoencoderTest("DEFAULTSERVER", 32, 16, 8, 2, 0.2);
    }

    @Test
    public void testAutoencoderParamservOutputs() {
        runAutoencoderTest("PARAMSERVER", 16, 8, 4, 1, DENSE);
    }
    private void runAutoencoderTest(String method, int h1, int h2, int h3, int maxEpochs, double sparsity) {
        int expectedHidden = h3 > 0 ? h3 : (h2 > 0 ? h2 : h1);
        loadTestConfiguration(getTestConfiguration(TEST_NAME));

        String home = SCRIPT_DIR + TEST_DIR;
        fullDMLScriptName = home + TEST_NAME + ".dml";
        programArgs = new String[] { "-nvargs",
                "X=" + input("X"),
                "H1=" + h1, "H2=" + h2, "H3=" + h3,
                "EPOCH=" + maxEpochs, "BATCH=" + 32,
                "STEP=" + 1e-4, "DECAY=" + 0.95, "MOMENTUM=" + 0.9,
                "METHOD=" + method, "MODE=LOCAL", "UTYPE=BSP",
                "FREQ=BATCH", "WORKERS=1", "SCHEME=DISJOINT_RANDOM",
                "NBATCHES=0", "MODELAVG=FALSE",
                "W1_out=" + output("W1_out"),
                "Wlast_out=" + output("Wlast_out"),
                "hidden_out=" + output("hidden_out")
        };

        double[][] X = getRandomMatrix(ROWS, COLS, 0, 1, sparsity, 42);
        writeInputMatrixWithMTD("X", X, true);

        runTest(true, false, null, -1);

        MatrixCharacteristics w1Meta = readDMLMetaDataFile("W1_out");
        MatrixCharacteristics wlastMeta = readDMLMetaDataFile("Wlast_out");
        MatrixCharacteristics hiddenMeta = readDMLMetaDataFile("hidden_out");

        Assert.assertEquals(h1, w1Meta.getRows());
        Assert.assertEquals(COLS, w1Meta.getCols());
        Assert.assertEquals(COLS, wlastMeta.getRows());
        Assert.assertEquals(h1, wlastMeta.getCols());
        Assert.assertEquals(ROWS, hiddenMeta.getRows());
        Assert.assertEquals(expectedHidden, hiddenMeta.getCols());
    }
}