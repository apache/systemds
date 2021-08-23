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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.functions.federated.FederatedTestObjectConstructor;
import org.junit.Assert;
import org.junit.Test;

@net.jcip.annotations.NotThreadSafe
public class FederatedReaderCSV extends AutomatedTestBase {

    private static final Log LOG = LogFactory.getLog(FederatedReaderCSV.class.getName());
    private final static String TEST_DIR = "functions/federated/ioR/";
    private final static String TEST_NAME = "FederatedReaderTest";
    private final static String TEST_CLASS_DIR = TEST_DIR + FederatedReaderCSV.class.getSimpleName() + "/";
    private final static int blocksize = 1024;

    private final static int dim = 3;
    long[][] begins = new long[][] {new long[] {0, 0}};
    long[][] ends = new long[][] {new long[] {dim, dim}};

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[] {"X1"}));
    }

    @Test
    public void testWithHeader() {
        federatedRead(true);
    }

    @Test
    public void testWithoutHeader() {
        federatedRead(false);
    }

    public void federatedRead( boolean header) {
        Types.ExecMode oldPlatform = setExecMode(ExecType.CP);
        getAndLoadTestConfiguration(TEST_NAME);
        setOutputBuffering(true);

        
        // empty script name because we don't execute any script, just start the worker
        
        fullDMLScriptName = "";
        int port1 = getRandomAvailablePort();
        Thread t1 = startLocalFedWorkerThread(port1);
        String host = "localhost";
        
        try {
            double[][] X1 = new double[][] {new double[] {1, 2, 3}, new double[] {4, 5, 6}, new double[] {7, 8, 9}};
            MatrixCharacteristics mc = new MatrixCharacteristics(dim, dim, blocksize, dim * dim);
            writeCSVMatrix("X1", X1, header, mc);

            // Thread.sleep(10000);
            MatrixObject fed = FederatedTestObjectConstructor.constructFederatedInput(dim, dim, blocksize, host, begins,
                ends, new int[] {port1}, new String[] {input("X1")}, input("X.json"));
            writeInputFederatedWithMTD("X.json", fed, null);

            // Run reference dml script with normal matrix

            fullDMLScriptName = SCRIPT_DIR + "functions/federated/io/" + TEST_NAME + "1Reference.dml";
            programArgs = new String[] {"-stats", "-args", input("X1")};

            String refOut = runTest(null).toString();

            LOG.debug(refOut);

            // Run federated
            fullDMLScriptName = SCRIPT_DIR + "functions/federated/io/" + TEST_NAME + ".dml";
            programArgs = new String[] {"-stats", "-args", input("X.json")};
            String out = runTest(null).toString();

            Assert.assertTrue(heavyHittersContainsString("fed_uak+"));
            // Verify output
            Assert.assertEquals(Double.parseDouble(refOut.split("\n")[0]), Double.parseDouble(out.split("\n")[0]),
                0.00001);
        }
        catch(Exception e) {
            e.printStackTrace();
            Assert.assertTrue(false);
        }
        finally {
            resetExecMode(oldPlatform);
        }

        TestUtils.shutdownThreads(t1);
    }
}
