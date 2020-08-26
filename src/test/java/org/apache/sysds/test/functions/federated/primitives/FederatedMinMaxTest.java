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

package org.apache.sysds.test.functions.federated.primitives;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;


@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedMinMaxTest extends AutomatedTestBase {
    private final static String TEST_DIR = "functions/federated/";
    private final static String TEST_NAME = "FederatedMinMaxTest";
    private final static String TEST_CLASS_DIR = TEST_DIR + FederatedMinMaxTest.class.getSimpleName() + "/";

    private final static int blocksize = 1024;
    @Parameterized.Parameter()
    public int rows;
    @Parameterized.Parameter(1)
    public int cols;
    @Parameterized.Parameter(2)
    public boolean rowPartitioned;

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
                {1000, 10, false}, {10, 1000, false},
                {1000, 10, true}, {10, 1000, true}, {1000, 1, true}
        });
    }

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"S", "R", "C"}));
    }

    @Test
//    @Ignore
    public void federatedMinMaxCP() { federatedMinMax(Types.ExecMode.SINGLE_NODE); }

    @Test
    public void federatedMinMaxSP() { federatedMinMax(Types.ExecMode.SPARK); }

    public void federatedMinMax(Types.ExecMode execMode) {
        boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
        Types.ExecMode platformOld = rtplatform;

        getAndLoadTestConfiguration(TEST_NAME);
        String HOME = SCRIPT_DIR + TEST_DIR;

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
        programArgs = new String[] {"-stats", "100", "-args", input("A"), input("A"), expected("R"), expected("C"), Boolean.toString(rowPartitioned).toUpperCase()};
        runTest(true, false, null, -1);

        // write expected
        double min = Arrays.stream(A).flatMapToDouble(Arrays::stream).min().getAsDouble();
        double max = Arrays.stream(A).flatMapToDouble(Arrays::stream).max().getAsDouble();

        writeExpectedMatrix("S", new double [][]{{min, max}});

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
                "rows=" + rows,
                "cols=" + cols,
                "rP=" + Boolean.toString(rowPartitioned).toUpperCase(),
                "out_S=" + output("S"),
                "out_R=" + output("R"),
                "out_C=" + output("C")};
        runTest(true, false, null, -1);

        // compare all sums via files
        compareResults(1e-9);

        Assert.assertTrue(heavyHittersContainsString("fed_uacmin"));
        Assert.assertTrue(heavyHittersContainsString("fed_uarmin"));
        Assert.assertTrue(heavyHittersContainsString("fed_uarmax"));
        Assert.assertTrue(heavyHittersContainsString("fed_uamin"));
        Assert.assertTrue(heavyHittersContainsString("fed_uamax"));

        TestUtils.shutdownThread(t);

        rtplatform = platformOld;
        DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
    }
}
