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
    private static final String TEST_NAME2 = "incSliceLineFull";
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
        addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }));
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
    public void testTop4HybridDPFullFewAdded() {
        runIncSliceLineTest(4, "e", true, false,2, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeDPFullFewAdded() {
        runIncSliceLineTest(4, "e", true, false,2, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop4HybridTPFullFewAdded() {
        runIncSliceLineTest(4, "e", false, false, 2, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeTPFullFewAdded() {
        runIncSliceLineTest(4, "e", false, false,2, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridDPFullFewAdded() {
        runIncSliceLineTest(10, "e", true, false,2, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeDPFullFewAdded() {
        runIncSliceLineTest(10, "e", true, false,2, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridTPFullFewAdded() {
        runIncSliceLineTest(10, "e", false, false,2, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeTPFullFewAdded() {
        runIncSliceLineTest(10, "e", false, false,2, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop4HybridDPSelFullFewAdded() {
        runIncSliceLineTest(4, "e", true, true,2, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeDPSelFullFewAdded() {
        runIncSliceLineTest(4, "e", true, true,2, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop4HybridTPSelFullFewAdded() {
        runIncSliceLineTest(4, "e", false, true,2, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeTPSelFullFewAdded() {
        runIncSliceLineTest(4, "e", false, true,4, false,  ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridDPSelFullFewAdded() {
        runIncSliceLineTest(10, "e", true, true, 2, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeDPSelFullFewAdded() {
        runIncSliceLineTest(10, "e", true, true, 1, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridTPSelFullFewAdded() {
        runIncSliceLineTest(10, "e", false, true, 2, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeTPSelFullFewAdded() {
        runIncSliceLineTest(10, "e", false, true, 2, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridTPSelE2FullFewAdded() {
        runIncSliceLineTest(10, "oe", false, true, 2, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeTPSelE2FullFewAdded() {
        runIncSliceLineTest(10, "oe", false, true, 2, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop4HybridDPFullManyAdded() {
        runIncSliceLineTest(4, "e", true, false,50, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeDPFullManyAdded() {
        runIncSliceLineTest(4, "e", true, false,50, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop4HybridTPFullManyAdded() {
        runIncSliceLineTest(4, "e", false, false, 50, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeTPFullManyAdded() {
        runIncSliceLineTest(4, "e", false, false,60, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridDPFullManyAdded() {
        runIncSliceLineTest(10, "e", true, false,50, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeDPFullManyAdded() {
        runIncSliceLineTest(10, "e", true, false,50, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridTPFullManyAdded() {
        runIncSliceLineTest(10, "e", false, false,90 , false, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeTPFullManyAdded() {
        runIncSliceLineTest(10, "e", false, false,99 , false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop4HybridDPSelFullManyAdded() {
        runIncSliceLineTest(4, "e", true, true,50, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeDPSelFullManyAdded() {
        runIncSliceLineTest(4, "e", true, true,50, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop4HybridTPSelFullManyAdded() {
        runIncSliceLineTest(4, "e", false, true,50, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeTPSelFullManyAdded() {
        runIncSliceLineTest(4, "e", false, true,50, false,  ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridDPSelFullManyAdded() {
        runIncSliceLineTest(10, "e", true, true, 50, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeDPSelFullManyAdded() {
        runIncSliceLineTest(10, "e", true, true, 50, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridTPSelFullManyAdded() {
        runIncSliceLineTest(10, "e", false, true, 50, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeTPSelFullManyAdded() {
        runIncSliceLineTest(10, "e", false, true, 50, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridTPSelE2FullManyAdded() {
        runIncSliceLineTest(10, "oe", false, true, 50, false, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeTPSelE2FullManyAdded() {
        runIncSliceLineTest(10, "oe", false, true, 50, false, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop4HybridDPFullFewAddedOnlyNull() {
        runIncSliceLineTest(4, "e", true, false,2, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeDPFullFewAddedOnlyNull() {
        runIncSliceLineTest(4, "e", true, false,2, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop4HybridTPFullFewAddedOnlyNull() {
        runIncSliceLineTest(4, "e", false, false, 2, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeTPFullFewAddedOnlyNull() {
        runIncSliceLineTest(4, "e", false, false,2, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridDPFullFewAddedOnlyNull() {
        runIncSliceLineTest(10, "e", true, false,2, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeDPFullFewAddedOnlyNull() {
        runIncSliceLineTest(10, "e", true, false,2, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridTPFullFewAddedOnlyNull() {
        runIncSliceLineTest(10, "e", false, false,2, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeTPFullFewAddedOnlyNull() {
        runIncSliceLineTest(10, "e", false, false,2, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop4HybridDPSelFullFewAddedOnlyNull() {
        runIncSliceLineTest(4, "e", true, true,2, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeDPSelFullFewAddedOnlyNull() {
        runIncSliceLineTest(4, "e", true, true,2, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop4HybridTPSelFullFewAddedOnlyNull() {
        runIncSliceLineTest(4, "e", false, true,2, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeTPSelFullFewAddedOnlyNull() {
        runIncSliceLineTest(4, "e", false, true,4, false,  ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridDPSelFullFewAddedOnlyNull() {
        runIncSliceLineTest(10, "e", true, true, 2, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeDPSelFullFewAddedOnlyNull() {
        runIncSliceLineTest(10, "e", true, true, 1, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridTPSelFullFewAddedOnlyNull() {
        runIncSliceLineTest(10, "e", false, true, 2, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeTPSelFullFewAddedOnlyNull() {
        runIncSliceLineTest(10, "e", false, true, 2, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridTPSelE2FullFewAddedOnlyNull() {
        runIncSliceLineTest(10, "oe", false, true, 2, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeTPSelE2FullFewAddedOnlyNull() {
        runIncSliceLineTest(10, "oe", false, true, 2, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop4HybridDPFullManyAddedOnlyNull() {
        runIncSliceLineTest(4, "e", true, false,50, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeDPFullManyAddedOnlyNull() {
        runIncSliceLineTest(4, "e", true, false,50, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop4HybridTPFullManyAddedOnlyNull() {
        runIncSliceLineTest(4, "e", false, false, 50, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeTPFullManyAddedOnlyNull() {
        runIncSliceLineTest(4, "e", false, false,60, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridDPFullManyAddedOnlyNull() {
        runIncSliceLineTest(10, "e", true, false,50, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeDPFullManyAddedOnlyNull() {
        runIncSliceLineTest(10, "e", true, false,50, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridTPFullManyAddedOnlyNull() {
        runIncSliceLineTest(10, "e", false, false,90 , true, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeTPFullManyAddedOnlyNull() {
        runIncSliceLineTest(10, "e", false, false,99 , true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop4HybridDPSelFullManyAddedOnlyNull() {
        runIncSliceLineTest(4, "e", true, true,50, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeDPSelFullManyAddedOnlyNull() {
        runIncSliceLineTest(4, "e", true, true,50, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop4HybridTPSelFullManyAddedOnlyNull() {
        runIncSliceLineTest(4, "e", false, true,50, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop4SinglenodeTPSelFullManyAddedOnlyNull() {
        runIncSliceLineTest(4, "e", false, true,50, false,  ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridDPSelFullManyAddedOnlyNull() {
        runIncSliceLineTest(10, "e", true, true, 50, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeDPSelFullManyAddedOnlyNull() {
        runIncSliceLineTest(10, "e", true, true, 50, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridTPSelFullManyAddedOnlyNull() {
        runIncSliceLineTest(10, "e", false, true, 50, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeTPSelFullManyAddedOnlyNull() {
        runIncSliceLineTest(10, "e", false, true, 50, true, ExecMode.SINGLE_NODE);
    }

    @Test
    public void testTop10HybridTPSelE2FullManyAddedOnlyNull() {
        runIncSliceLineTest(10, "oe", false, true, 50, true, ExecMode.HYBRID);
    }

    @Test
    public void testTop10SinglenodeTPSelE2FullManyAddedOnlyNull() {
        runIncSliceLineTest(10, "oe", false, true, 50, true, ExecMode.SINGLE_NODE);
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
        double[][] oldX = {
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
        };
        double[][] addedX = {       
                { 4, 2, 4, 4, 2, 1, 2, 1, 1, 4 },
                { 4, 1, 1, 4, 1, 4, 3, 2, 4, 2 },
                { 2, 1, 2, 2, 3, 1, 4, 3, 3, 4 },
                { 4, 1, 3, 1, 3, 1, 2, 1, 3, 3 },
                { 2, 1, 3, 1, 1, 3, 1, 2, 1, 2 },
                { 1, 3, 4, 3, 1, 2, 2, 4, 1, 1 },
                { 2, 4, 4, 3, 4, 1, 2, 1, 2, 4 },
                { 3, 3, 3, 3, 3, 1, 2, 3, 4, 4 },
                { 3, 2, 2, 2, 4, 1, 4, 2, 3, 1 },
                { 1, 2, 3, 2, 4, 3, 2, 3, 2, 3 }
        };
        double[][] oldE = {
                { 0.159 }, { 0.588 }, { 0.414 }, { 0.305 }, { 0.193 }, { 0.195 }, { 0.878 }, { 0.149 }, { 0.835 },
                { 0.344 },
                { 0.123 }, { 0.456 }, { 0.789 }, { 0.987 }, { 0.654 }, { 0.321 }, { 0.246 }, { 0.135 }, { 0.579 },
                { 0.802 },
                { 0.591 }, { 0.858 }, { 0.144 }, { 0.350 }, { 0.931 }, { 0.951 }, { 0.788 }, { 0.491 }, { 0.358 },
                { 0.443 },
        };
        double[][] addedE = {
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

        testIncSliceLineCustomInputsFull(addedX, oldX, oldE, addedE, K, correctRes);
    }


    @Test
    public void testIncSliceLineCustomInputsFull() {
        double[][] newX = {
            {1, 1, 1, 1},
            {1, 2, 2, 2},
            {1, 3, 3, 3},
            {1, 4, 4, 4},
            {5, 2, 5, 5},
            {6, 2, 6, 6},
            {7, 2, 7, 7},
            {8, 2, 8, 8},
            {9, 9, 9, 9},
            {1, 1, 1, 1},
            {2, 2, 2, 2},
            {3, 3, 3, 3},
            {4, 4, 4, 4},
            {5, 5, 5, 5},
            {6, 6, 6, 6},
            {7, 7, 7, 7},
            {8, 8, 8, 8},
            {9, 9, 9, 9},
            {1, 1, 1, 1},
            {2, 2, 2, 2},
            {3, 3, 3, 3},
            {4, 4, 4, 4},
            {5, 5, 5, 5},
            {6, 6, 6, 6},
            {7, 7, 7, 7},
            {8, 8, 8, 8},
            {9, 9, 9, 9},
            {1, 1, 1, 1},
            {2, 2, 2, 2},
            {3, 3, 3, 3},
            {4, 4, 4, 4},
            {5, 5, 5, 5},
            {6, 6, 6, 6},
            {7, 7, 7, 7},
            {8, 8, 8, 8},
            {9, 9, 9, 9},
            {10, 10, 10, 10},
            {11, 11, 11, 11},
            {12, 12, 12, 12},
            {13, 13, 13, 13},
            {14, 14, 14, 14},
            {15, 15, 15, 15},
            {16, 16, 16, 16},
            {17, 17, 17, 17},
            {18, 18, 18, 18},
            {19, 19, 19, 19},
            {20, 20, 20, 20},
            {10, 10, 10, 10},
            {11, 11, 11, 11},
            {12, 12, 12, 12},
            {13, 13, 13, 13},
            {14, 14, 14, 14},
            {15, 15, 15, 15},
            {16, 16, 16, 16},
            {17, 17, 17, 17},
            {18, 18, 18, 18},
            {19, 19, 19, 19},
            {20, 20, 20, 20},
            {10, 10, 10, 10},
            {11, 11, 11, 11},
            {12, 12, 12, 12},
            {13, 13, 13, 13},
            {14, 14, 14, 14},
            {15, 15, 15, 15},
            {16, 16, 16, 16},
            {17, 17, 17, 17},
            {18, 18, 18, 18},
            {19, 19, 19, 19},
            {20, 20, 20, 20},
            {10, 10, 10, 10},
            {11, 11, 11, 11},
            {12, 12, 12, 12},
            {13, 13, 13, 13},
            {14, 14, 14, 14},
            {15, 15, 15, 15},
            {16, 16, 16, 20},
            {17, 17, 17, 20},
            {18, 18, 18, 20},
            {19, 19, 19, 20},
            {20, 20, 20, 20}
        };

        double[][] e = {
                {0.001}, {0.002}, {0.003}, {0.004}, {0.005}, {0.006}, {0.007}, {0.008}, {0.009}, {0.010},
                {0.011}, {0.012}, {0.013}, {0.014}, {0.015}, {0.016}, {0.017}, {0.018}, {0.019}, {0.020},
                {0.021}, {0.022}, {0.023}, {0.024}, {0.025}, {0.026}, {0.027}, {0.028}, {0.029}, {0.030},
                {0.031}, {0.032}, {0.033}, {0.034}, {0.035}, {0.036}, {0.037}, {0.038}, {0.039}, {0.040},
                {0.041}, {0.042}, {0.043}, {0.044}, {0.045}, {0.046}, {0.047}, {0.048}, {0.049}, {0.050},
                {0.051}, {0.052}, {0.053}, {0.054}, {0.055}, {0.056}, {0.057}, {0.058}, {0.059}, {0.060},
                {0.061}, {0.062}, {0.063}, {0.064}, {0.065}, {0.066}, {0.067}, {0.068}, {0.069}, {0.070},
                {0.071}, {0.072}, {0.073}, {0.074}, {0.075}, {0.076}, {0.077}, {0.078}, {0.079}, {0.080}
                
        };

        runIncSliceLineTest(newX, e, 10, "e", false, true, 50, false, ExecMode.SINGLE_NODE);
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

    private void runIncSliceLineTest(int K, String err, boolean dp, boolean selCols, int proportionOfTuplesAddedInPercent, boolean onlyNullEAdded, ExecMode mode) {
        runIncSliceLineTest(null, null, K, err, dp, selCols, proportionOfTuplesAddedInPercent, onlyNullEAdded, mode);
    }


    private void runIncSliceLineTest(double[][] customX, double[][] customE,int K, String err, boolean dp, boolean selCols, int proportionOfTuplesAddedInPercent, boolean onlyNullEAdded, ExecMode mode) {
     
        ExecMode platformOld = setExecMode(mode);
        loadTestConfiguration(getTestConfiguration(TEST_NAME2));
        String HOME = SCRIPT_DIR + TEST_DIR;
        String data = DATASET_DIR + "Salaries.csv";

        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME2));

            
            double[][] newX = null;
            double[][] e = null;
            // read output and store for dml and R
            if(customX != null && customE != null){
                newX = customX;
                e = customE;
            } else {
                // run data preparation
                fullDMLScriptName = HOME + PREP_NAME + ".dml";
                programArgs = new String[] { "-args", data, err, output("newX"), output("e") };
                runTest(true, false, null, -1);
            
                newX = TestUtils.convertHashMapToDoubleArray(readDMLMatrixFromOutputDir("newX"));
                e = TestUtils.convertHashMapToDoubleArray(readDMLMatrixFromOutputDir("e"));
            }
            int numOfAddedTuples = (int) Math.round(newX.length * proportionOfTuplesAddedInPercent / 100.0);

            double[][] addedX = new double[numOfAddedTuples][newX[0].length];
            double[][] oldX = new double[newX.length - numOfAddedTuples][newX[0].length];

            for (int i = 0; i < numOfAddedTuples; i++) {
                addedX[i] = newX[i];
            }

            for (int i = numOfAddedTuples; i < newX.length; i++) {
                oldX[i - numOfAddedTuples] = newX[i];
            }
            double[][] addedE = new double[numOfAddedTuples][e[0].length];
            double[][] oldE = new double[e.length - numOfAddedTuples][e[0].length];
            if(onlyNullEAdded){
                for (int i = 0; i < numOfAddedTuples; i++) {
                    addedE[i][0] = 0;
                    e[i][0] = 0;
                }
            } else {
                for (int i = 0; i < numOfAddedTuples; i++) {
                    addedE[i] = e[i];
                }
            }
            
            for (int i = numOfAddedTuples; i < e.length; i++) {
                oldE[i - numOfAddedTuples] = e[i];
            }

            writeInputMatrixWithMTD("addedX", addedX, false);
            writeInputMatrixWithMTD("oldX", oldX, false);
            writeInputMatrixWithMTD("oldE", oldE, false);
            writeInputMatrixWithMTD("addedE", addedE, false);

            fullDMLScriptName = HOME + TEST_NAME2 + ".dml";
            programArgs = new String[] { "-args", input("addedX"), input("oldX"), input("oldE"), input("addedE"), String.valueOf(K),
                    String.valueOf(!dp).toUpperCase(), String.valueOf(selCols).toUpperCase(),
                    String.valueOf(VERBOSE).toUpperCase(), output("R1"), output("R2") };

            runTest(true, false, null, -1);

            HashMap<CellIndex, Double> dmlfile1 = readDMLMatrixFromOutputDir("R1");
            HashMap<CellIndex, Double> dmlfile2 = readDMLMatrixFromOutputDir("R2");
            double[][] ret1 = TestUtils.convertHashMapToDoubleArray(dmlfile1);
            double[][] ret2 = TestUtils.convertHashMapToDoubleArray(dmlfile2);

            TestUtils.compareMatrices(ret1, ret2, 1e-2);

            
            if(customX != null && customE != null){
                newX = customX;
                e = customE;
            } 
            // execute main test
            writeInputMatrixWithMTD("newX", newX, false);
            writeInputMatrixWithMTD("e", e, false);            
            fullDMLScriptName = HOME + "slicefinder" + ".dml";
            programArgs = new String[] { "-args", input("newX"), input("e"), String.valueOf(K),
                    String.valueOf(!dp).toUpperCase(), String.valueOf(selCols).toUpperCase(),
                    String.valueOf(VERBOSE).toUpperCase(), output("R") };

            runTest(true, false, null, -1);

            HashMap<CellIndex, Double> dmlfile3 = readDMLMatrixFromOutputDir("R");


            TestUtils.compareMatrices(dmlfile1, dmlfile3, 1e-2, "R1", "R");
            

            // compare expected results
            if (err.equals("e") && customX == null && customE == null && !onlyNullEAdded) {
                double[][] ret = TestUtils.convertHashMapToDoubleArray(dmlfile1);
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

    public void testIncSliceLineCustomInputsFull(double[][] addedX, double[][] oldX, double[][] oldE, double[][] addedE, int K, double[][] correctRes) {
        boolean dp = true, selCols = false;
        ExecMode mode = ExecMode.SINGLE_NODE;
        ExecMode platformOld = setExecMode(mode);
        loadTestConfiguration(getTestConfiguration(TEST_NAME2));
        String HOME = SCRIPT_DIR + TEST_DIR;

        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME2));

            writeInputMatrixWithMTD("addedX", addedX, false);
            writeInputMatrixWithMTD("oldX", oldX, false);
            writeInputMatrixWithMTD("oldE", oldE, false);
            writeInputMatrixWithMTD("addedE", addedE, false);

            fullDMLScriptName = HOME + TEST_NAME2 + ".dml";
            programArgs = new String[] { "-args", input("addedX"), input("oldX"), input("oldE"), input("addedE"), String.valueOf(K),
                    String.valueOf(!dp).toUpperCase(), String.valueOf(selCols).toUpperCase(),
                    String.valueOf(VERBOSE).toUpperCase(), output("R1"), output("R2") };

            runTest(true, false, null, -1);

            HashMap<CellIndex, Double> dmlfile1 = readDMLMatrixFromOutputDir("R1");
            HashMap<CellIndex, Double> dmlfile2 = readDMLMatrixFromOutputDir("R2");
            double[][] ret1 = TestUtils.convertHashMapToDoubleArray(dmlfile1);
            double[][] ret2 = TestUtils.convertHashMapToDoubleArray(dmlfile2);
            TestUtils.compareMatrices(correctRes, ret2, 1e-2);
            TestUtils.compareMatrices(ret1, ret2, 1e-2);

            Assert.assertFalse(heavyHittersContainsSubString("evalSlice"));
        } finally {
            rtplatform = platformOld;
        }
    }

}