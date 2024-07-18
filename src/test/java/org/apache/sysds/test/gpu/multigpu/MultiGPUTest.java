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

package org.apache.sysds.test.gpu.multigpu;

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

public class MultiGPUTest extends AutomatedTestBase {
    private static final String TEST_DIR = "gpu/";
    private static final String TEST_CLASS_DIR = TEST_DIR + MultiGPUTest.class.getSimpleName() +  "/";
    private static final String SINGLE_GPU_TEST = "SingleGPUTest";
    private static final String MULTI_GPUS_TEST = "MultiGPUsTest";
    private static final String TEST_NAME = "GPUTest";
    private static final String DATA_SET = DATASET_DIR + "MNIST/mnist_test.csv";

    @Override
    public void setUp() {
        TEST_GPU = true;
        VERBOSE_STATS = true;
        addTestConfiguration(SINGLE_GPU_TEST,
                new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"R"}));
        addTestConfiguration(MULTI_GPUS_TEST,
                new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"R"}));
        TestConfiguration singleConfig = availableTestConfigurations.get(SINGLE_GPU_TEST);
        singleConfig.addVariable("sysds.gpu.availableGPUs", 0);
        TestConfiguration multiConfig = availableTestConfigurations.get(MULTI_GPUS_TEST);
        multiConfig.addVariable("sysds.gpu.availableGPUs", -1);
    }

    @Test
    public void SingleGPUTest() {
        runMultiGPUsTest(false);
    }

    @Test
    public void MultiGPUsTest() {
        runMultiGPUsTest(true);
    }

    /**
     * Run the test with multiple GPUs
     * @param multiGPUs whether to run the test with multiple GPUs
     */
    private void runMultiGPUsTest(boolean multiGPUs) {
        getAndLoadTestConfiguration(multiGPUs ? MULTI_GPUS_TEST : SINGLE_GPU_TEST);

        String HOME = SCRIPT_DIR + TEST_DIR;
        fullDMLScriptName = HOME + TEST_NAME + ".dml";
        programArgs = new String[]{"-args", DATA_SET, output("R") };
        fullRScriptName = HOME + TEST_NAME + ".R";

        rCmd = null;

        runTest(true, false, null, -1);
    }
}
