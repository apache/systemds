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
package org.apache.sysds.test.functions.federated.io;

import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedWriterTest extends AutomatedTestBase {

    // private static final Log LOG = LogFactory.getLog(FederatedWriterTest.class.getName());
    private final static String TEST_DIR = "functions/federated/";
    private final static String TEST_NAME = "FederatedWriterTest";
    private final static String TEST_CLASS_DIR = TEST_DIR + FederatedWriterTest.class.getSimpleName() + "/";
    private final static int blocksize = 1024;

    @Parameterized.Parameter()
    public int rows;
    @Parameterized.Parameter(1)
    public int cols;
    @Parameterized.Parameter(2)
    public boolean rowPartitioned;
    @Parameterized.Parameter(3)
    public int fedCount;

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
    }

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        // number of rows or cols has to be >= number of federated locations.
        return Arrays.asList(new Object[][] {{10, 13, true, 2},});
    }

    @Test
    public void federatedSinglenodeWrite() {
        federatedWrite(Types.ExecMode.SINGLE_NODE);
    }

    public void federatedWrite(Types.ExecMode execMode) {
        boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
        Types.ExecMode platformOld = rtplatform;
        rtplatform = execMode;
        if(rtplatform == Types.ExecMode.SPARK) {
            DMLScript.USE_LOCAL_SPARK_CONFIG = true;
        }
        getAndLoadTestConfiguration(TEST_NAME);

        // write input matrices
        int halfRows = rows / 2;
        // We have two matrices handled by a single federated worker
        double[][] X1 = getRandomMatrix(halfRows, cols, 0, 1, 1, 42);
        double[][] X2 = getRandomMatrix(halfRows, cols, 0, 1, 1, 1340);
        writeInputMatrixWithMTD("X1", X1, false, new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
        writeInputMatrixWithMTD("X2", X2, false, new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
        // empty script name because we don't execute any script, just start the worker
        fullDMLScriptName = "";
        int port1 = getRandomAvailablePort();
        int port2 = getRandomAvailablePort();
        Thread t1 = startLocalFedWorkerThread(port1);
        Thread t2 = startLocalFedWorkerThread(port2);

        try {

            // Run reader and write a federated json to enable the rest of the test
            fullDMLScriptName = SCRIPT_DIR + "functions/federated/io/FederatedReaderTestCreate.dml";
            programArgs = new String[] {"-stats", "-explain","-args", input("X1"), input("X2"), port1 + "", port2 + "", input("X.json")};
            // String writer = runTest(null).toString();
            runTest(null);

            // Run reference dml script with normal matrix
            fullDMLScriptName = SCRIPT_DIR + "functions/federated/io/FederatedReaderTest.dml";
            programArgs = new String[] {"-stats", "-args", input("X.json")};
            String out = runTest(null).toString();

            Assert.assertTrue(heavyHittersContainsString("fed_uak+"));

            fullDMLScriptName = SCRIPT_DIR + "functions/federated/io/FederatedReference.dml";
            // programArgs = new String[] {"-args", input("X1"), input("X2")};
            programArgs = new String[] {"-stats", "100", "-nvargs",
                "in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
                "in_X2=" + TestUtils.federatedAddress(port2, input("X2")), "rows=" + rows, "cols=" + cols};
            String refOut = runTest(null).toString();

            // Run federated

            // Verify output
            Assert.assertEquals(Double.parseDouble(refOut.split("\n")[0]),
                Double.parseDouble(out.split("\n")[0]),
                0.00001);
        }
        catch(Exception e) {
            e.printStackTrace();
            Assert.assertTrue(false);
        }

        TestUtils.shutdownThreads(t1, t2);
        rtplatform = platformOld;
        DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
    }
}
