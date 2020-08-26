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

package org.apache.sysds.test.functions.federated.indexing;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedRightIndexTest extends AutomatedTestBase {
    private final static String TEST_DIR = "functions/federated/";
    private final static String TEST_NAME = "FederatedRightIndexTest";
    private final static String TEST_CLASS_DIR = TEST_DIR + FederatedRightIndexTest.class.getSimpleName() + "/";

    private final static int blocksize = 1024;
    @Parameterized.Parameter()
    public int rows;
    @Parameterized.Parameter(1)
    public int cols;

    @Parameterized.Parameter(2)
    public int from;

    @Parameterized.Parameter(3)
    public int to;

    @Parameterized.Parameter(4)
    public boolean rowPartitioned;

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
                {10, 1000,  1, 7, true},
//                {50, 1000, 11, 49, true}
        });
    }

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R", "C", "S"}));
    }

    @Test
    public void federatedRightIndexCP() {
        federatedRightIndex(Types.ExecMode.SINGLE_NODE);
    }

    @Test
    public void federatedRightIndexSP() {
        federatedRightIndex(Types.ExecMode.SPARK);
    }

    public void federatedRightIndex(Types.ExecMode execMode) {
        boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
        Types.ExecMode platformOld = rtplatform;

        getAndLoadTestConfiguration(TEST_NAME);
        String HOME = SCRIPT_DIR + TEST_DIR;
//        setOutputBuffering(false);

        int r = rows;
        int c = cols / 2;
        if (rowPartitioned) {
            r = rows / 2;
            c = cols;
        }
        double[][] A = getRandomMatrix(r, c, -10, 10, 1, 1);
        writeInputMatrixWithMTD("A", A, false, new MatrixCharacteristics(r, c, blocksize, (r / 2) * c));
        int port = getRandomAvailablePort();
        Process t = startLocalFedWorker(port);

        // we need the reference file to not be written to hdfs, so we get the correct format
        rtplatform = Types.ExecMode.SINGLE_NODE;

        // Run reference dml script with normal matrix for Row/Col
        fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
        programArgs = new String[] {"-stats", "100", "-args", input("A"), input("A"), String.valueOf(from), String.valueOf(to),  Boolean.toString(rowPartitioned).toUpperCase(),
                expected("R"), expected("C"), expected("S")};
        runTest(true, false, null, -1);

        // reference file should not be written to hdfs, so we set platform here
        rtplatform = execMode;
        if(rtplatform == Types.ExecMode.SPARK) {
            DMLScript.USE_LOCAL_SPARK_CONFIG = true;
        }
        TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
        loadTestConfiguration(config);
        fullDMLScriptName = HOME + TEST_NAME + ".dml";
        programArgs = new String[] {"-stats", "100", "-nvargs", "in=" +
                TestUtils.federatedAddress(port, input("A")),
                "rows=" + rows, "cols=" + cols,
                "from=" + from, "to=" + to, "rP=" + Boolean.toString(rowPartitioned).toUpperCase(),
                "out_S=" + output("S"),
                "out_R=" + output("R"),
                "out_C=" + output("C")};

        runTest(true, false, null, -1);

        // compare all sums via files
        compareResults(1e-11);

        TestUtils.shutdownThread(t);
        rtplatform = platformOld;
        DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
    }
}
