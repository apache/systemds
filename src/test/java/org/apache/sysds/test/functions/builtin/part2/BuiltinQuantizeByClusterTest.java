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
@net.jcip.annotations.NotThreadSafe
public class BuiltinQuantizeByClusterTest extends AutomatedTestBase {

	@Parameter
	public String test_case;
	@Parameter(1)
	public int rows;
	@Parameter(2)
	public int cols;
	@Parameter(3)
	public int clusters;
	@Parameter(4)
	public int subspaces;
	@Parameter(5)
	public int k;
	@Parameter(6)
	public int vectors_per_cluster;
	@Parameter(7)
	public boolean quantize_separately;
	public boolean space_decomp;

	private final static String TEST_NAME = "quantizeByCluster";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinQuantizeByClusterTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	private final static int runs = 3;
	private final static int max_iter = 1000;

	@Parameterized.Parameters(name = "{0}: rows={1}, cols={2}, c={3}, subspaces={4}, k={5}, v_per_c={6}, sep={7}")
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][]{
			{"sub_cluster", 1024, 64, 12, 8, 12, 40, true},  {"sub_cluster", 1024, 64, 12, 4, 12, 40, true},  {"sub_cluster", 1024, 64, 12, 2, 12, 40, true},
			{"sub_cluster", 1024, 64, 12, 8, 12, 40, false}, {"sub_cluster", 1024, 64, 12, 4, 12, 40, false}, {"sub_cluster", 1024, 64, 12, 2, 12, 40, false},
			{"cluster", 1024, 64, 12, 8, 12, 40, true},  {"cluster", 1024, 64, 12, 4, 12, 40, true},  {"cluster", 1024, 64, 12, 2, 12, 40, true},
			{"cluster", 1024, 64, 20, 8, 12, 40, false}, {"cluster", 1024, 64, 12, 4, 12, 40, false}, {"cluster", 1024, 64, 12, 2, 12, 40, false},
			{"uniform", 1024, 64, 12, 8, 12, 40, true},  {"uniform", 1024, 64, 12, 4, 12, 40, true},  {"uniform", 1024, 64, 12, 2, 12, 40, true},
			{"uniform", 1024, 64, 12, 8, 12, 40, false}, {"uniform", 1024, 64, 12, 4, 12, 40, false}, {"uniform", 1024, 64, 12, 2, 12, 40, false},
			{"normal",  1024, 64, 12, 8, 12, 40, true},  {"normal",  1024, 64, 12, 4, 12, 40, true},  {"normal",  1024, 64, 12, 2, 12, 40, true},
			{"normal",  1024, 64, 12, 8, 12, 40, false}, {"normal",  1024, 64, 12, 4, 12, 40, false}, {"normal",  1024, 64, 12, 2, 12, 40, false},
			{"normal",  1024, 53, 12, 8, 12, 40, true},  {"normal",  1024, 61, 12, 4, 12, 40, true},  {"normal",  1024, 83, 12, 2, 12, 40, true},
			{"normal",  1024, 53, 12, 8, 12, 40, false},  {"normal",  1024, 61, 12, 4, 12, 40, false},  {"normal",  1024, 83, 12, 2, 12, 40, false},
		});
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
	}

	@Test
	public void basicTest() {
		for (boolean b : new boolean[]{true, false}) {
			space_decomp = b;
			runQuantizeByClusterTest();
		}
	}

	/*The tests use kmeans clustering as a baseline and check whether the distortion is within
	a certain threshold.*/
	private void runQuantizeByClusterTest() {
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-nvargs", "codes=" + output("codes"), "codebook=" + output("codebook"),
			"pq_distortion=" + output("pq_distortion"), "k_distortion=" + output("k_distortion"), "is_orthogonal=" + output("is_orthogonal"),
			"clusters=" + clusters, "test_case=" + test_case, "rows=" + rows,
			"cols=" + cols, "subspaces=" + subspaces, "k=" + k, "runs=" + runs, "max_iter=" + max_iter,
			"eps=" + eps, "vectors_per_cluster=" + vectors_per_cluster, "sep=" + quantize_separately, "space_decomp=" + space_decomp};

		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

		//check if R is an orthogonal matrix
		if(space_decomp) {
			double is_orthogonal = readDMLScalarFromOutputDir("is_orthogonal").get(new MatrixValue.CellIndex(1, 1));
			Assert.assertEquals(0, is_orthogonal, 0.001);
		}

		//check if output dimensions are correct
		MatrixCharacteristics meta_codes = readDMLMetaDataFile("codes");
		MatrixCharacteristics meta_codebook = readDMLMetaDataFile("codebook");
		Assert.assertTrue("Matrix dimensions should be equal to expected dimensions",
			meta_codes.getRows() == (long) clusters * vectors_per_cluster 
			&& meta_codes.getCols() == subspaces);
		Assert.assertEquals("Centroid dimensions should be equal to expected dimensions",
				(int) Math.ceil((double) cols / subspaces), meta_codebook.getCols());

		//check if distortion is within a threshold
		double pq_distortion = readDMLScalarFromOutputDir("pq_distortion").get(new MatrixValue.CellIndex(1, 1));
		double k_distortion = readDMLScalarFromOutputDir("k_distortion").get(new MatrixValue.CellIndex(1, 1));
		if (!test_case.equals("cluster")) {
			Assert.assertTrue(pq_distortion < 1.2 * k_distortion + 0.5);
		} else {
			Assert.assertTrue(pq_distortion < 2.4 * k_distortion + 2);
		}
	}
}
