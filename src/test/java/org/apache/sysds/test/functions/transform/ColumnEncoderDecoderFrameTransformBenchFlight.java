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

package org.apache.sysds.test.functions.transform;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.utils.Statistics;
import org.junit.Test;

public class ColumnEncoderDecoderFrameTransformBenchFlight extends AutomatedTestBase {

    private static final String TEST_NAME = "flightFTBenchmark";
    private static final String TEST_DIR = "functions/transform/";
    private static final String TEST_CLASS_DIR = TEST_DIR + ColumnEncoderDecoderFrameTransformBenchFlight.class.getSimpleName() + "/";

    // Dataset & spec paths
    private static final String DATASET = "flight/flight.csv";
    private static final String SPEC = "flight/kdd_spec1.json";
    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME,
                new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"R"}));
    }

    @Test
    public void runFrameTransformBenchmark() {
        runBenchmark(ExecMode.SINGLE_NODE,"csv");
    }

    private void runBenchmark(ExecMode execMode, String ofmt) {
        ExecMode oldMode = rtplatform;
        rtplatform = execMode;

        boolean oldSparkConfig = DMLScript.USE_LOCAL_SPARK_CONFIG;
        if (execMode == ExecMode.SPARK || execMode == ExecMode.HYBRID) {
            DMLScript.USE_LOCAL_SPARK_CONFIG = true;
        }

        try {
            // Load configuration (creates temp dirs etc.)
            getAndLoadTestConfiguration(TEST_NAME);

            // Setup script
            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[]{
                    "-stats",
                    "-args",
                    DATASET_DIR + DATASET,
                    DATASET_DIR + SPEC,
                    output("R")
            };

            System.out.println("\n===== Running Flight Benchmark =====");
            Statistics.reset();
            long start = System.nanoTime();

            // Execute benchmark
            runTest(true, false, null, -1);

            long end = System.nanoTime();
            double elapsed = (end - start) / 1e9;
            System.out.printf("Execution Time: %.3f seconds%n", elapsed);

        } catch (Exception ex) {
            throw new RuntimeException(ex);
        } finally {
            rtplatform = oldMode;
            DMLScript.USE_LOCAL_SPARK_CONFIG = oldSparkConfig;
        }
    }
}
