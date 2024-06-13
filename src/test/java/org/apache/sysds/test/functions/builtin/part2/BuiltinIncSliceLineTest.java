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

package org.apache.sysds.test.functions.builtin.part2;

import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class BuiltinIncSliceLineTest extends AutomatedTestBase {
    private static final String PREP_NAME = "slicefinderPrep";
    private static final String TEST_NAME = "incSliceLine";
    private static final String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinIncSliceLineTest.class.getSimpleName() + "/";
    private static final boolean VERBOSE = true;

    private static final double[][] EXPECTED_TOPK = new double[][] {
            { 1.042, 69210699988.477, 11078019685.642, 18.000 },
            { 0.478, 92957580467.849, 11078019685.642, 39.000 },
            { 0.316, 40425449547.480, 11078019685.642, 10.000 },
            { 0.262, 67630559163.266, 7261504482.540, 29.000 },
            { 0.224, 202448990843.317, 11119010986.000, 125.000 },
            { 0.218, 68860581248.568, 7261504482.540, 31.000 },
            { 0.164, 206527445340.279, 11119010986.000, 135.000 },
            { 0.122, 68961886413.866, 7261504482.540, 34.000 },
            { 0.098, 360278523220.479, 11119010986.000, 266.000 },
            { 0.092, 73954209826.485, 11078019685.642, 39.000 }
    };

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "R" }));
    }

    @Test
    public void testTop4HybridDP() {
        runIncSliceLineTest(4, "e", true, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeDP() {
        runIncSliceLineTest(4, "e", true, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop4HybridTP() {
        runIncSliceLineTest(4, "e", false, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeTP() {
        runIncSliceLineTest(4, "e", false, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridDP() {
        runIncSliceLineTest(10, "e", true, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeDP() {
        runIncSliceLineTest(10, "e", true, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridTP() {
        runIncSliceLineTest(10, "e", false, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeTP() {
        runIncSliceLineTest(10, "e", false, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop4HybridDPSel() {
        runIncSliceLineTest(4, "e", true, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeDPSel() {
        runIncSliceLineTest(4, "e", true, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop4HybridTPSel() {
        runIncSliceLineTest(4, "e", false, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeTPSel() {
        runIncSliceLineTest(4, "e", false, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridDPSel() {
        runIncSliceLineTest(10, "e", true, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeDPSel() {
        runIncSliceLineTest(10, "e", true, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridTPSel() {
        runIncSliceLineTest(10, "e", false, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeTPSel() {
        runIncSliceLineTest(10, "e", false, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridTPSelE2() {
        runIncSliceLineTest(10, "oe", false, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeTPSelE2() {
        runIncSliceLineTest(10, "oe", false, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testIncSliceLineCustomInputs1() {
        double[][] newX = {
                { 2, 1, 1, 2, 3, 2, 3, 3, 1, 2 },
                { 2, 2, 2, 3, 4, 1, 2, 1, 3, 2 },
                { 2, 1, 3, 3, 2, 2, 3, 1, 1, 4 },
                { 1, 2, 2, 1, 3, 2, 3, 2, 2, 3 },
                { 3, 2, 3, 4, 3, 3, 4, 1, 1, 3 },
                { 4, 3, 2, 3, 4, 4, 3, 4, 1, 1 },
                { 2, 2, 2, 4, 3, 3, 2, 2, 1, 2 },
                { 1, 1, 2, 2, 3, 3, 2, 1, 1, 2 },
                { 4, 3, 2, 1, 3, 2, 4, 2, 4, 3 },
                { 1, 3, 1, 4, 1, 3, 3, 2, 3, 2 },
                { 2, 4, 3, 1, 2, 4, 1, 3, 2, 4 },
                { 3, 2, 4, 3, 1, 4, 2, 3, 4, 1 },
                { 4, 1, 2, 4, 3, 1, 4, 2, 1, 3 },
                { 1, 3, 4, 2, 4, 3, 1, 4, 2, 3 },
                { 2, 4, 1, 3, 2, 4, 3, 1, 4, 2 },
                { 3, 2, 4, 1, 3, 4, 2, 3, 1, 4 },
                { 4, 1, 3, 2, 4, 1, 4, 2, 3, 1 },
                { 1, 3, 2, 4, 1, 3, 4, 2, 4, 3 },
                { 2, 4, 1, 3, 2, 4, 3, 1, 2, 4 },
                { 2, 3, 3, 2, 1, 4, 2, 3, 2, 3 }
        };
        double[][] e = {
                { 0.159 }, { 0.588 }, { 0.414 }, { 0.305 }, { 0.193 }, { 0.195 }, { 0.878 }, { 0.149 }, { 0.835 },
                { 0.344 },
                { 0.123 }, { 0.456 }, { 0.789 }, { 0.987 }, { 0.654 }, { 0.321 }, { 0.246 }, { 0.135 }, { 0.579 },
                { 0.802 }
        };
        int K = 10;
        double[][] correctRes = {
                { 0.307, 2.807, 0.878, 4.000 },
                { 0.307, 2.807, 0.878, 4.000 },
                { 0.282, 2.759, 0.987, 4.000 },
                { 0.157, 4.046, 0.987, 7.000 },
                { 0.127, 2.956, 0.878, 5.000 },
                { 0.122, 2.942, 0.878, 5.000 },
                { 0.074, 3.298, 0.987, 6.000 },
                { 0.064, 4.197, 0.878, 8.000 },
                { 0.061, 2.796, 0.987, 5.000 },
                { 0.038, 3.194, 0.878, 6.000 }
        };
        testIncSliceLineCustomInputs(newX, e, K, correctRes);
    }

    @Test
    public void testIncSliceLineCustomInputs2() {
        double[][] newX = {
                { 2, 1, 1, 1, 3, 4, 2, 2, 1, 2 },
                { 3, 3, 3, 2, 1, 2, 3, 1, 4, 2 },
                { 3, 2, 3, 1, 1, 1, 4, 3, 4, 2 },
                { 1, 3, 2, 3, 2, 3, 2, 1, 2, 1 },
                { 4, 3, 1, 1, 1, 1, 1, 1, 3, 2 },
                { 2, 2, 3, 3, 2, 2, 2, 3, 4, 1 },
                { 3, 2, 2, 2, 4, 4, 2, 4, 1, 1 },
                { 1, 3, 3, 2, 1, 3, 1, 2, 4, 4 },
                { 2, 1, 2, 2, 3, 1, 2, 3, 2, 1 },
                { 4, 1, 3, 4, 1, 4, 2, 3, 4, 4 },
                { 4, 2, 4, 4, 2, 1, 2, 1, 1, 4 },
                { 4, 1, 1, 4, 1, 4, 3, 2, 4, 2 },
                { 2, 1, 2, 2, 3, 1, 4, 3, 3, 4 },
                { 4, 1, 3, 1, 3, 1, 2, 1, 3, 3 },
                { 2, 1, 3, 1, 1, 3, 1, 2, 1, 2 },
                { 1, 3, 4, 3, 1, 2, 2, 2, 1, 1 },
                { 2, 4, 4, 3, 4, 1, 2, 1, 2, 4 },
                { 3, 3, 3, 3, 3, 1, 2, 3, 4, 4 },
                { 3, 2, 2, 2, 4, 1, 4, 2, 3, 1 },
                { 1, 2, 3, 2, 4, 3, 2, 3, 2, 3 }
        };

        double[][] e = {
                { 0.591 }, { 0.858 }, { 0.144 }, { 0.350 }, { 0.931 }, { 0.951 }, { 0.788 }, { 0.491 }, { 0.358 },
                { 0.443 },
                { 0.231 }, { 0.564 }, { 0.897 }, { 0.879 }, { 0.546 }, { 0.132 }, { 0.462 }, { 0.153 }, { 0.759 },
                { 0.028 }
        };
        int K = 10;
        double[][] correctRes = {
                { 0.410, 3.466, 0.931, 4.000 },
                { 0.410, 3.466, 0.931, 4.000 },
                { 0.111, 2.802, 0.897, 4.000 },
                { 0.075, 3.805, 0.951, 6.000 },
                { 0.057, 4.278, 0.897, 7.000 },
                { 0.047, 3.711, 0.931, 6.000 },
                { 0.035, 3.152, 0.897, 5.000 },
                { 0.032, 4.179, 0.897, 7.000 },
                { 0.023, 3.634, 0.931, 6.000 },
                { 0.013, 3.091, 0.931, 5.000 }
        };

        testIncSliceLineCustomInputs(newX, e, K, correctRes);
    }

    @Test
    public void testIncSliceLineCustomInputs3() {
        double[][] newX = {
                { 2, 1, 1, 2, 3, 2, 3, 3, 1, 2 },
                { 2, 2, 2, 3, 4, 1, 2, 1, 3, 2 },
                { 2, 1, 3, 3, 2, 2, 3, 1, 1, 4 },
                { 1, 2, 2, 1, 3, 2, 3, 2, 2, 3 },
                { 3, 2, 3, 4, 3, 3, 4, 1, 1, 3 },
                { 4, 3, 2, 3, 4, 4, 3, 4, 1, 1 },
                { 2, 2, 2, 4, 3, 3, 2, 2, 1, 2 },
                { 1, 1, 2, 2, 3, 3, 2, 1, 1, 2 },
                { 4, 3, 2, 1, 3, 2, 4, 2, 4, 3 },
                { 1, 3, 1, 4, 1, 3, 3, 2, 3, 2 },
                { 2, 4, 3, 1, 2, 4, 1, 3, 2, 4 },
                { 3, 2, 4, 3, 1, 4, 2, 3, 4, 1 },
                { 4, 1, 2, 4, 3, 1, 4, 2, 1, 3 },
                { 1, 3, 4, 2, 4, 3, 1, 4, 2, 3 },
                { 2, 4, 1, 3, 2, 4, 3, 1, 4, 2 },
                { 3, 2, 4, 1, 3, 4, 2, 3, 1, 4 },
                { 4, 1, 3, 2, 4, 1, 4, 2, 3, 1 },
                { 1, 3, 2, 4, 1, 3, 4, 2, 4, 3 },
                { 2, 4, 1, 3, 2, 4, 3, 1, 2, 4 },
                { 2, 3, 3, 2, 1, 4, 2, 3, 2, 3 },
                { 2, 1, 1, 1, 3, 4, 2, 2, 1, 2 },
                { 3, 3, 3, 2, 1, 2, 3, 1, 4, 2 },
                { 3, 2, 3, 1, 1, 1, 4, 3, 4, 2 },
                { 1, 3, 2, 3, 2, 3, 2, 1, 2, 1 },
                { 4, 3, 1, 1, 1, 1, 1, 1, 3, 2 },
                { 2, 2, 3, 3, 2, 2, 2, 3, 4, 1 },
                { 3, 2, 2, 2, 4, 4, 2, 4, 1, 1 },
                { 1, 3, 3, 2, 1, 3, 1, 2, 4, 4 },
                { 2, 1, 2, 2, 3, 1, 2, 3, 2, 1 },
                { 4, 1, 3, 4, 1, 4, 2, 3, 4, 4 },
                { 4, 2, 4, 4, 2, 1, 2, 1, 1, 4 },
                { 4, 1, 1, 4, 1, 4, 3, 2, 4, 2 },
                { 2, 1, 2, 2, 3, 1, 4, 3, 3, 4 },
                { 4, 1, 3, 1, 3, 1, 2, 1, 3, 3 },
                { 2, 1, 3, 1, 1, 3, 1, 2, 1, 2 },
                { 1, 3, 4, 3, 1, 2, 2, 2, 1, 1 },
                { 2, 4, 4, 3, 4, 1, 2, 1, 2, 4 },
                { 3, 3, 3, 3, 3, 1, 2, 3, 4, 4 },
                { 3, 2, 2, 2, 4, 1, 4, 2, 3, 1 },
                { 1, 2, 3, 2, 4, 3, 2, 3, 2, 3 }
        };
        double[][] e = {
                { 0.159 }, { 0.588 }, { 0.414 }, { 0.305 }, { 0.193 }, { 0.195 }, { 0.878 }, { 0.149 }, { 0.835 },
                { 0.344 },
                { 0.123 }, { 0.456 }, { 0.789 }, { 0.987 }, { 0.654 }, { 0.321 }, { 0.246 }, { 0.135 }, { 0.579 },
                { 0.802 },
                { 0.591 }, { 0.858 }, { 0.144 }, { 0.350 }, { 0.931 }, { 0.951 }, { 0.788 }, { 0.491 }, { 0.358 },
                { 0.443 },
                { 0.231 }, { 0.564 }, { 0.897 }, { 0.879 }, { 0.546 }, { 0.132 }, { 0.462 }, { 0.153 }, { 0.759 },
                { 0.028 }
        };
        int K = 10;
        double[][] correctRes = {
                { 0.149, 4.300, 0.931, 6.000 },
                { 0.113, 3.138, 0.987, 4.000 },
                { 0.093, 4.644, 0.931, 7.000 },
                { 0.090, 4.630, 0.951, 7.000 },
                { 0.059, 8.002, 0.951, 14.000 },
                { 0.024, 2.954, 0.951, 4.000 },
                { 0.017, 3.415, 0.897, 5.000 },
                { 0.010, 3.398, 0.878, 5.000 },
                { 0.009, 2.923, 0.897, 4.000 },
                { 0.008, 3.391, 0.897, 5.000 }
        };
        testIncSliceLineCustomInputs(newX, e, K, correctRes);
    }

    @Test
    public void testIncSliceLineCustomInputs4() {
        double[][] newX = {
                { 2, 1, 1, 2, 3, 2, 3, 3, 1, 2 },
                { 2, 2, 2, 3, 4, 1, 2, 1, 3, 2 },
                { 2, 1, 3, 3, 2, 2, 3, 1, 1, 4 },
                { 1, 2, 2, 1, 3, 2, 3, 2, 2, 3 },
                { 3, 2, 3, 4, 3, 3, 4, 1, 1, 3 },
                { 4, 3, 2, 3, 4, 4, 3, 4, 1, 1 },
                { 2, 2, 2, 4, 3, 3, 2, 2, 1, 2 },
                { 1, 1, 2, 2, 3, 3, 2, 1, 1, 2 },
                { 4, 3, 2, 1, 3, 2, 4, 2, 4, 3 },
                { 1, 3, 1, 4, 1, 3, 3, 2, 3, 2 },
                { 2, 4, 3, 1, 2, 4, 1, 3, 2, 4 },
                { 3, 2, 4, 3, 1, 4, 2, 3, 4, 1 },
                { 4, 1, 2, 4, 3, 1, 4, 2, 1, 3 },
                { 1, 3, 4, 2, 4, 3, 1, 4, 2, 3 },
                { 2, 4, 1, 3, 2, 4, 3, 1, 4, 2 },
                { 3, 2, 4, 1, 3, 4, 2, 3, 1, 4 },
                { 4, 1, 3, 2, 4, 1, 4, 2, 3, 1 },
                { 1, 3, 2, 4, 1, 3, 4, 2, 4, 3 },
                { 2, 4, 1, 3, 2, 4, 3, 1, 2, 4 },
                { 2, 3, 3, 2, 1, 4, 2, 3, 2, 3 },
                { 2, 1, 1, 1, 3, 4, 2, 2, 1, 2 },
                { 3, 3, 3, 2, 1, 2, 3, 1, 4, 2 },
                { 3, 2, 3, 1, 1, 1, 4, 3, 4, 2 },
                { 1, 3, 2, 3, 2, 3, 2, 1, 2, 1 },
                { 4, 3, 1, 1, 1, 1, 1, 1, 3, 2 },
                { 2, 2, 3, 3, 2, 2, 2, 3, 4, 1 },
                { 3, 2, 2, 2, 4, 4, 2, 4, 1, 1 },
                { 1, 3, 3, 2, 1, 3, 1, 2, 4, 4 },
                { 2, 1, 2, 2, 3, 1, 2, 3, 2, 1 },
                { 4, 1, 3, 4, 1, 4, 2, 3, 4, 4 },
                { 4, 2, 4, 4, 2, 1, 2, 1, 1, 4 },
                { 4, 1, 1, 4, 1, 4, 3, 2, 4, 2 },
                { 2, 1, 2, 2, 3, 1, 4, 3, 3, 4 },
                { 4, 1, 3, 1, 3, 1, 2, 1, 3, 3 },
                { 2, 1, 3, 1, 1, 3, 1, 2, 1, 2 },
                { 1, 3, 4, 3, 1, 2, 2, 2, 1, 1 },
                { 2, 4, 4, 3, 4, 1, 2, 1, 2, 4 },
                { 3, 3, 3, 3, 3, 1, 2, 3, 4, 4 },
                { 3, 2, 2, 2, 4, 1, 4, 2, 3, 1 },
                { 1, 2, 3, 2, 4, 3, 2, 3, 2, 3 }
        };
        double[][] e = {
                { 0.159 }, { 0.588 }, { 0.414 }, { 0.305 }, { 0.193 }, { 0.195 }, { 0.878 }, { 0.149 }, { 0.835 },
                { 0.344 },
                { 0.123 }, { 0.456 }, { 0.789 }, { 0.987 }, { 0.654 }, { 0.321 }, { 0.246 }, { 0.135 }, { 0.579 },
                { 0.802 },
                { 0.591 }, { 0.858 }, { 0.144 }, { 0.350 }, { 0.931 }, { 0.951 }, { 0.788 }, { 0.491 }, { 0.358 },
                { 0.443 },
                { 0.231 }, { 0.564 }, { 0.897 }, { 0.879 }, { 0.546 }, { 0.132 }, { 0.462 }, { 0.153 }, { 0.759 },
                { 0.028 }
        };

        int K = 10;

        double[][] correctRes = {
                { 0.149, 4.300, 0.931, 6.000 },
                { 0.113, 3.138, 0.987, 4.000 },
                { 0.093, 4.644, 0.931, 7.000 },
                { 0.090, 4.630, 0.951, 7.000 },
                { 0.059, 8.002, 0.951, 14.000 },
                { 0.024, 2.954, 0.951, 4.000 },
                { 0.017, 3.415, 0.897, 5.000 },
                { 0.010, 3.398, 0.878, 5.000 },
                { 0.009, 2.923, 0.897, 4.000 },
                { 0.008, 3.391, 0.897, 5.000 }
        };

        double[][] m1 = {
            {1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {4.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 3.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 3.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 3.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 4.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 2.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 3.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 4.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 3.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 4.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 3.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 4.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 3.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 4.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 3.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 4.000}
        };

        double[][] m2 = {
            {2.000, 2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {2.000, 3.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {2.000, 0.000, 2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {2.000, 0.000, 0.000, 0.000, 3.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {2.000, 0.000, 0.000, 0.000, 0.000, 4.000, 0.000, 0.000, 0.000, 0.000},
            {2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000, 0.000, 0.000},
            {2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000, 0.000},
            {2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000},
            {2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000},
            {2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000},
            {2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 3.000},
            {0.000, 2.000, 2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 2.000, 0.000, 0.000, 3.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 2.000, 0.000, 0.000, 0.000, 4.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 2.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000, 0.000, 0.000},
            {0.000, 2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000, 0.000},
            {0.000, 2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000},
            {0.000, 2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000},
            {0.000, 2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000},
            {0.000, 2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 3.000},
            {0.000, 3.000, 2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 3.000, 0.000, 0.000, 3.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 3.000, 0.000, 0.000, 0.000, 4.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 3.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000, 0.000, 0.000},
            {0.000, 3.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000, 0.000},
            {0.000, 3.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000},
            {0.000, 3.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000},
            {0.000, 3.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000},
            {0.000, 3.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 3.000},
            {0.000, 0.000, 2.000, 0.000, 3.000, 0.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 2.000, 0.000, 0.000, 4.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 2.000, 0.000, 0.000, 0.000, 2.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 2.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000, 0.000},
            {0.000, 0.000, 2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000},
            {0.000, 0.000, 2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000},
            {0.000, 0.000, 2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000},
            {0.000, 0.000, 2.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 3.000},
            {0.000, 0.000, 0.000, 0.000, 3.000, 4.000, 0.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 3.000, 0.000, 2.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 3.000, 0.000, 0.000, 2.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 3.000, 0.000, 0.000, 0.000, 1.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 3.000, 0.000, 0.000, 0.000, 2.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 3.000, 0.000, 0.000, 0.000, 0.000, 2.000},
            {0.000, 0.000, 0.000, 0.000, 3.000, 0.000, 0.000, 0.000, 0.000, 3.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 4.000, 2.000, 0.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 4.000, 0.000, 2.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 4.000, 0.000, 0.000, 1.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 4.000, 0.000, 0.000, 2.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 4.000, 0.000, 0.000, 0.000, 2.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 4.000, 0.000, 0.000, 0.000, 3.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 2.000, 0.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000, 1.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000, 2.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000, 0.000, 2.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000, 0.000, 3.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 1.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 2.000, 0.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000, 2.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000, 3.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 2.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 3.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 2.000},
            {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 3.000}
        };
        
        double[][] m3 = {{0.000, 0.000, 2.000, 0.000, 3.000, 0.000, 0.000, 2.000, 0.000, 0.000}};

        double[][] r1 = {
            {-0.303, 1.920, 0.987, 5.000},
            {0.064, 4.197, 0.878, 8.000},
            {-0.078, 2.065, 0.835, 4.000},
            {-0.370, 1.757, 0.789, 5.000},
            {-0.118, 2.741, 0.878, 6.000},
            {0.074, 3.298, 0.987, 6.000},
            {-0.249, 1.736, 0.654, 4.000},
            {-0.020, 3.874, 0.878, 8.000},
            {-0.362, 1.778, 0.802, 5.000},
            {-0.328, 1.584, 0.835, 4.000},
            {-0.127, 2.343, 0.987, 5.000},
            {-0.068, 2.886, 0.654, 6.000},
            {-0.129, 2.339, 0.878, 5.000},
            {-0.248, 1.737, 0.802, 4.000},
            {-0.231, 1.770, 0.654, 4.000},
            {-0.083, 3.629, 0.878, 8.000},
            {-0.104, 2.016, 0.987, 4.000},
            {-0.261, 1.713, 0.835, 4.000},
            {-0.137, 2.686, 0.987, 6.000},
            {-0.115, 3.130, 0.802, 7.000},
            {0.038, 3.194, 0.878, 6.000},
            {-0.257, 2.650, 0.654, 7.000},
            {-0.187, 2.198, 0.835, 5.000},
            {-0.175, 2.577, 0.654, 6.000},
            {0.005, 3.532, 0.878, 7.000},
            {-0.327, 1.861, 0.802, 5.000},
            {-0.221, 3.098, 0.878, 8.000},
            {0.061, 2.796, 0.987, 5.000},
            {-0.070, 2.080, 0.835, 4.000},
            {-0.108, 2.772, 0.878, 6.000},
            {0.157, 4.046, 0.987, 7.000},
            {-0.404, 1.437, 0.579, 4.000}
        };

        double[][] r2 = {
            {0.122, 1.466, 0.878, 2.000},
            {-0.235, 0.802, 0.802, 1.000},
            {0.122, 1.466, 0.878, 2.000},
            {-0.324, 1.037, 0.878, 2.000},
            {-0.030, 2.158, 0.802, 4.000},
            {0.336, 2.268, 0.878, 3.000},
            {-0.077, 0.878, 0.878, 1.000},
            {-0.229, 1.451, 0.878, 3.000},
            {-0.193, 1.504, 0.802, 3.000},
            {0.033, 2.279, 0.878, 4.000},
            {-0.235, 0.802, 0.802, 1.000},
            {-0.008, 1.771, 0.878, 3.000},
            {-0.269, 1.697, 0.878, 4.000},
            {-0.593, 0.777, 0.456, 2.000},
            {0.014, 2.243, 0.878, 4.000},
            {-0.172, 1.183, 0.878, 2.000},
            {-0.270, 1.392, 0.878, 3.000},
            {-1.267, 0.305, 0.305, 1.000},
            {0.122, 1.466, 0.878, 2.000},
            {-0.883, 0.498, 0.305, 2.000},
            {-0.427, 1.165, 0.835, 3.000},
            {-0.166, 0.835, 0.835, 1.000},
            {-0.365, 0.997, 0.802, 2.000},
            {-0.235, 0.802, 0.802, 1.000},
            {-0.324, 1.314, 0.835, 3.000},
            {-1.495, 0.195, 0.195, 1.000},
            {0.457, 1.789, 0.987, 2.000},
            {-1.186, 0.344, 0.344, 1.000},
            {0.282, 2.759, 0.987, 4.000},
            {0.127, 2.956, 0.878, 5.000},
            {-1.495, 0.195, 0.195, 1.000},
            {-0.116, 1.615, 0.878, 3.000},
            {0.122, 2.942, 0.878, 5.000},
            {-0.106, 2.011, 0.878, 4.000},
            {-1.267, 0.305, 0.305, 1.000},
            {-0.116, 1.615, 0.878, 3.000},
            {-0.079, 2.064, 0.835, 4.000},
            {-1.234, 0.321, 0.321, 1.000},
            {-0.300, 1.348, 0.878, 3.000},
            {0.307, 2.807, 0.878, 4.000},
            {-0.205, 2.489, 0.878, 6.000},
            {-1.267, 0.305, 0.305, 1.000},
            {-0.413, 1.186, 0.878, 3.000},
            {-0.049, 2.122, 0.835, 4.000},
            {-0.141, 1.579, 0.802, 3.000},
            {-1000000000.00, 0.000, 0.000, 0.000},
            {-0.864, 0.516, 0.321, 2.000},
            {-0.193, 1.504, 0.802, 3.000},
            {-0.542, 0.654, 0.654, 1.000},
            {-0.235, 0.802, 0.802, 1.000},
            {-0.077, 0.878, 0.878, 1.000},
            {-0.300, 1.348, 0.878, 3.000},
            {-0.235, 0.802, 0.802, 1.000},
            {-0.116, 1.615, 0.878, 3.000},
            {-0.235, 0.802, 0.802, 1.000},
            {0.330, 1.667, 0.878, 2.000},
            {-1.267, 0.305, 0.305, 1.000},
            {-0.132, 1.222, 0.878, 2.000},
            {-0.079, 2.064, 0.835, 4.000},
            {-0.413, 1.186, 0.878, 3.000},
            {-0.381, 0.982, 0.789, 2.000},
            {-10000000.00, 0.000, 0.000, 0.000},
            {0.216, 2.094, 0.987, 3.000}
        };
        
        double[][] r3 = {{0.307, 2.807, 0.878, 4.000}};

        testIncSliceLineCustomInputs(newX, e, m1, m2, m3, r1, r2, r3, K, correctRes);
    }

    // @Test
    // public void testTop10SparkTP() {
    // runIncSliceLineTest(10, false, ExecMode.SPARK);
    // }

    private void runIncSliceLineTest(int K, String err, boolean dp, boolean selCols, ExecMode mode) {
        ExecMode platformOld = setExecMode(mode);
        loadTestConfiguration(getTestConfiguration(TEST_NAME));
        String HOME = SCRIPT_DIR + TEST_DIR;
        String data = DATASET_DIR + "Salaries.csv";

        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME));

            // run data preparation
            fullDMLScriptName = HOME + PREP_NAME + ".dml";
            programArgs = new String[] { "-args", data, err, output("newX"), output("e") };
            runTest(true, false, null, -1);

            // read output and store for dml and R
            double[][] newX = TestUtils.convertHashMapToDoubleArray(readDMLMatrixFromOutputDir("newX"));
            double[][] e = TestUtils.convertHashMapToDoubleArray(readDMLMatrixFromOutputDir("e"));
            writeInputMatrixWithMTD("newX", newX, true);
            writeInputMatrixWithMTD("e", e, true);

            // execute main test
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[] { "-args", input("newX"), input("e"), String.valueOf(K),
                    String.valueOf(!dp).toUpperCase(), String.valueOf(selCols).toUpperCase(),
                    String.valueOf(VERBOSE).toUpperCase(), output("R") };

            runTest(true, false, null, -1);

            HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");

            // execute main test
            fullDMLScriptName = HOME + "slicefinder" + ".dml";
            programArgs = new String[] { "-args", input("newX"), input("e"), String.valueOf(K),
                    String.valueOf(!dp).toUpperCase(), String.valueOf(selCols).toUpperCase(),
                    String.valueOf(VERBOSE).toUpperCase(), output("R") };

            runTest(true, false, null, -1);

            HashMap<CellIndex, Double> dmlfile2 = readDMLMatrixFromOutputDir("R");

            TestUtils.compareMatrices(dmlfile, dmlfile2, 1e-2, "Stat-IncSliceLine", "Stat-Slicefinder");

            // compare expected results
            if (err.equals("e")) {
                double[][] ret = TestUtils.convertHashMapToDoubleArray(dmlfile);
                if (mode != ExecMode.SPARK) // TODO why only CP correct, but R always matches? test framework?
                    for (int i = 0; i < K; i++)
                        TestUtils.compareMatrices(EXPECTED_TOPK[i], ret[i], 1e-2);
            }

            // ensure proper inlining, despite initially multiple calls and large function
            Assert.assertFalse(heavyHittersContainsSubString("evalSlice"));
        } finally {
            rtplatform = platformOld;
        }
    }

    public void testIncSliceLineCustomInputs(double[][] newX, double[][] e, int K, double[][] correctRes) {
        boolean dp = true, selCols = false;
        ExecMode mode = ExecMode.SINGLE_NODE;
        ExecMode platformOld = setExecMode(mode);
        loadTestConfiguration(getTestConfiguration(TEST_NAME));
        String HOME = SCRIPT_DIR + TEST_DIR;

        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME));

            writeInputMatrixWithMTD("newX", newX, false);
            writeInputMatrixWithMTD("e", e, false);

            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[] { "-args", input("newX"), input("e"), String.valueOf(K),
                    String.valueOf(!dp).toUpperCase(), String.valueOf(selCols).toUpperCase(),
                    String.valueOf(VERBOSE).toUpperCase(), output("R") };

            runTest(true, false, null, -1);

            HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
            double[][] ret = TestUtils.convertHashMapToDoubleArray(dmlfile);
            TestUtils.compareMatrices(correctRes, ret, 1e-2);

            Assert.assertFalse(heavyHittersContainsSubString("evalSlice"));
        } finally {
            rtplatform = platformOld;
        }
    }


    public void testIncSliceLineCustomInputs(double[][] newX, double[][] e, double[][] m1, double[][] m2, 
    double[][] m3, double[][] r1, double[][] r2, double[][] r3, int K, double[][] correctRes) {
    boolean dp = true, selCols = false;
    ExecMode mode = ExecMode.SINGLE_NODE;
    ExecMode platformOld = setExecMode(mode);
    loadTestConfiguration(getTestConfiguration(TEST_NAME));
    String HOME = SCRIPT_DIR + TEST_DIR;

    try {
        loadTestConfiguration(getTestConfiguration(TEST_NAME));

        // Write input matrices
        writeInputMatrixWithMTD("newX", newX, false);
        writeInputMatrixWithMTD("e", e, false);
        writeInputMatrixWithMTD("m1", m1, false);
        writeInputMatrixWithMTD("m2", m2, false);
        writeInputMatrixWithMTD("m3", m3, false);
        writeInputMatrixWithMTD("r1", r1, false);
        writeInputMatrixWithMTD("r2", r2, false);
        writeInputMatrixWithMTD("r3", r3, false);

        // Set DML script and arguments
        fullDMLScriptName = HOME + TEST_NAME + ".dml";
        programArgs = new String[] { 
            "-args", 
            input("newX"), 
            input("e"), 
            String.valueOf(K),
            String.valueOf(!dp).toUpperCase(), 
            String.valueOf(selCols).toUpperCase(),
            String.valueOf(VERBOSE).toUpperCase(), 
            output("R"),
            
            input("m1"), 
            input("m2"),
            input("m3"), 
            input("r1"), 
            input("r2"),
            input("r3")

            
        };


        // Run the test
        runTest(true, false, null, -1);

        // Read and compare the output
        HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
        double[][] ret = TestUtils.convertHashMapToDoubleArray(dmlfile);
        TestUtils.compareMatrices(correctRes, ret, 1e-2);

        // Ensure proper inlining
        Assert.assertFalse(heavyHittersContainsSubString("evalSlice"));
    } catch (Exception ex) {
        ex.printStackTrace();
        Assert.fail("Exception occurred during test execution: " + ex.getMessage());
    } finally {
        rtplatform = platformOld;
    }
}

}