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

package org.apache.sysds.test.functions.compress;

import static org.junit.Assert.assertTrue;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.DMLCompressionStatistics;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

public class compressInstruction extends AutomatedTestBase {

    protected String getTestClassDir() {
        return getTestDir() + this.getClass().getSimpleName() + "/";
    }

    protected String getTestName() {
        return "compress";
    }

    protected String getTestDir() {
        return "functions/compress/compressInstruction/";
    }

    @Test
    public void empty() {

    }

    @Test
    public void testCompressInstruction_01() {
        compressTest(1, 1000, 0.2, ExecType.CP, 0, 5, 0, 1, "01");
    }

    @Test
    public void testCompressInstruction_02() {
        compressTest(1, 1000, 0.2, ExecType.CP, 0, 5, 1, 1, "02");
    }

    public void compressTest(int cols, int rows, double sparsity, LopProperties.ExecType instType, int min, int max,
        int decompressionCountExpected, int compressionCountsExpected, String name) {

        Types.ExecMode platformOld = setExecMode(instType);
        try {

            loadTestConfiguration(getTestConfiguration(getTestName()));

            fullDMLScriptName = SCRIPT_DIR + "/" + getTestDir() + "compress_" + name + ".dml";

            programArgs = new String[] {"-stats", "100", "-nvargs", "cols=" + cols, "rows=" + rows,
                "sparsity=" + sparsity, "min=" + min, "max= " + max};

            runTest(null);

            int decompressCount = 0;
            decompressCount += DMLCompressionStatistics.getDecompressionCount();
            decompressCount += DMLCompressionStatistics.getDecompressionSTCount();
            long compressionCount = Statistics.getCPHeavyHitterCount("compress");

            Assert.assertEquals(compressionCount, compressionCountsExpected);
            Assert.assertEquals(decompressionCountExpected, decompressCount);

        }
        catch(Exception e) {
            e.printStackTrace();
            assertTrue("Exception in execution: " + e.getMessage(), false);
        }
        finally {
            rtplatform = platformOld;
        }
    }

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(getTestName(), new TestConfiguration(getTestClassDir(), getTestName()));
    }

}
