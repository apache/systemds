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

import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;

@RunWith(Parameterized.class)
public class BuiltinQuantizeByClusterTest extends AutomatedTestBase {

    @Parameter public String test_case;
    @Parameter(1) public int rows;
    @Parameter(2) public int cols;
    @Parameter(3) public int clusters;
    @Parameter(4) public int subvector_size;
    @Parameter(5) public int k;
    @Parameter(6) public int runs;
    @Parameter(7) public int max_iter;
    @Parameter(8) public int vectors_per_cluster;
    @Parameter(9) public boolean quantize_separately;

    private final static String TEST_NAME = "quantizeByCluster";
    private final static String TEST_DIR = "functions/builtin/";
    private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinQuantizeByClusterTest.class.getSimpleName() + "/";
    private final static double eps = 1e-10;
    private final static double cluster_offset = 0.1;

    @Parameterized.Parameters(name = "{0}: rows={1}, cols={2}, c={3}, subv_size={4}, k={5}, runs={6}, max_iter={7}, v_per_c={8}, sep={9}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
                        {"sub_cluster", 1024, 64, 12, 8, 12, 5, 1000, 40, true}, {"sub_cluster", 1024, 64, 6, 4, 6, 5, 1000, 40, true}, {"sub_cluster", 1024, 64, 3, 2, 3, 5, 1000, 40, true},
                        {"sub_cluster", 1024, 64, 12, 8, 12, 5, 1000, 40, false}, {"sub_cluster", 1024, 64, 12, 4, 12, 5, 1000, 40, false}, {"sub_cluster", 1024, 64, 12, 2, 12, 5, 1000, 40, false},
                        {"cluster", 1024, 64, 12, 8, 12, 5, 1000, 40, true}, {"cluster", 1024, 64, 6, 4, 20, 5, 1000, 40, true}, {"cluster", 1024, 64, 3, 2, 3, 5, 1000, 40, true},
                        {"cluster", 1024, 64, 20, 8, 12, 5, 1000, 40, false}, {"cluster", 1024, 64, 20, 4, 20, 5, 1000, 40, false}, {"cluster", 1024, 64, 12, 2, 12, 5, 1000, 40, false},
                        {"uniform", 1024, 64, 12, 8, 12, 5, 1000, 40, true}, {"uniform", 1024, 64, 6, 4, 20, 5, 1000, 40, true}, {"uniform", 1024, 64, 3, 2, 3, 5, 1000, 40, true},
                        {"uniform", 1024, 64, 12, 8, 12, 5, 1000, 40, false}, {"uniform", 1024, 64, 12, 4, 12, 5, 1000, 40, false}, {"uniform", 1024, 64, 12, 2, 12, 5, 1000, 40, false},
                        {"normal", 1024, 64, 12, 8, 12, 5, 1000, 40, true}, {"normal", 1024, 64, 6, 4, 6, 5, 1000, 40, true}, {"normal", 1024, 64, 3, 2, 3, 5, 1000, 40, true},
                        {"normal", 1024, 64, 12, 8, 12, 5, 1000, 40, false}, {"normal", 1024, 64, 12, 4, 12, 5, 1000, 40, false}, {"normal", 1024, 64, 12, 2, 12, 5, 1000, 40, false},
                }
        );
    }

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
    }

    @Test
    public void basicTest() {
        runQuantizeByClusterTest();
    }

    private void runQuantizeByClusterTest() {

        loadTestConfiguration(getTestConfiguration(TEST_NAME));
        String HOME = SCRIPT_DIR + TEST_DIR;
        fullDMLScriptName = HOME + TEST_NAME + ".dml";
        programArgs = new String[]{"-nvargs", "codes=" + output("codes"), "codebook=" + output("codebook"),
                "pq_distortion=" + output("pq_distortion"), "k_distortion=" + output("k_distortion"),
                "clusters=" + clusters, "test_case=" + test_case, "rows=" + rows,
                "cols=" + cols, "subvector_size=" + subvector_size, "k=" + k, "runs=" + runs, "max_iter=" + max_iter,
                "eps=" + eps, "vectors_per_cluster=" + vectors_per_cluster, "sep=" + quantize_separately};

        runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);


        // check if output dimensions are correct
        MatrixCharacteristics meta_codes = readDMLMetaDataFile("codes");
        MatrixCharacteristics meta_codebook = readDMLMetaDataFile("codebook");
        Assert.assertTrue("Matrix dimensions should be equal to expected dimensions",
                meta_codes.getRows() == clusters * vectors_per_cluster && meta_codes.getCols() == cols / subvector_size);
        Assert.assertEquals("Centroid dimensions should be equal to expected dimensions", meta_codebook.getCols(),
                subvector_size);

        double pq_distortion = readDMLMatrixFromOutputDir("pq_distortion").get(new MatrixValue.CellIndex(1, 1));
        double k_distortion = readDMLMatrixFromOutputDir("k_distortion").get(new MatrixValue.CellIndex(1, 1));

        if (!test_case.equals("cluster")) {
            Assert.assertTrue(pq_distortion < 1.2 * k_distortion);
        } else {
            Assert.assertTrue(pq_distortion < 20);
        }

    }
}
