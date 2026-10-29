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

package org.apache.sysds.test.applications;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class PageRankTest extends AutomatedTestBase {

	protected final static String TEST_DIR = "applications/page_rank/";
	protected final static String TEST_NAME = "PageRank";
	protected String TEST_CLASS_DIR = TEST_DIR + PageRankTest.class.getSimpleName() + "/";

	protected int numRows, numCols;

	public PageRankTest(int rows, int cols) {
		this.numRows = rows;
		this.numCols = cols;
	}

	@Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][] { { 50, 50 }, { 1500, 1500 }, { 7500, 7500 }, {10000,10000} };
		return Arrays.asList(data);
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "p" }));
	}

	@Test
	public void testPageRank() {
		System.out.println("------------ BEGIN " + TEST_NAME + " TEST {" + numRows + ", "
				+ numCols + "} ------------");

		int rows = numRows;
		int cols = numCols;
		int maxiter = 2;
		double alpha = 0.85;

		getAndLoadTestConfiguration(TEST_NAME);
		
		List<String> proArgs = new ArrayList<>();
		proArgs.add("-args");
		proArgs.add(input("g"));
		proArgs.add(input("p"));
		proArgs.add(input("e"));
		proArgs.add(input("u"));
		proArgs.add(Double.toString(alpha));
		proArgs.add(Integer.toString(maxiter));
		proArgs.add(output("p"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		
		fullDMLScriptName = getScript();

		double[][] g = getRandomMatrix(rows, cols, 1, 1, 0.0000042159, -1);
		double[][] p = getRandomMatrix(rows, 1, 1, 1, 1, -1);
		double[][] e = getRandomMatrix(rows, 1, 1, 1, 1, -1);
		double[][] u = getRandomMatrix(1, cols, 1, 1, 1, -1);
		writeInputMatrixWithMTD("g", g, true);
		writeInputMatrixWithMTD("p", p, true);
		writeInputMatrixWithMTD("e", e, true);
		writeInputMatrixWithMTD("u", u, true);

		for (int i = 0; i < maxiter; i++) {
			double[][] gp = TestUtils.performMatrixMultiplication(g, p);
			double[][] eu = TestUtils.performMatrixMultiplication(e, u);
			double[][] eup = TestUtils.performMatrixMultiplication(eu, p);
			for (int j = 0; j < rows; j++) {
				p[j][0] = alpha * gp[j][0] + (1 - alpha) * eup[j][0];
			}
		}

		writeExpectedMatrix("p", p);

		/*
		 * Expected number of jobs: Reblock - 1 job While loop iteration - 6 jobs (can be reduced if MMCJ job can
		 * produce multiple outputs) Final output write - 1 job
		 */
		int expectedNumberOfJobs = 8;
		runTest(true, EXCEPTION_NOT_EXPECTED, null, expectedNumberOfJobs);

		compareResults(0.0000001);
	}
	/*
	* Twitter2010 edge list is local and not part of this repository.
	*/
	@Test
	@Ignore
	public void testPageRankTwitter2010() throws IOException {
		// run once; this class is parameterized over several sizes
		if (numRows != 50 || numCols != 50) {
			return;
		}

		int maxiter = 2;
		double alpha = 0.85;
		int maxNodes = Integer.getInteger("sysds.test.twitter2010.maxNodes", 500);
		int maxLines = Integer.getInteger("sysds.test.twitter2010.maxLines", 2000000);

		getAndLoadTestConfiguration(TEST_NAME);
			
		List<String> proArgs = new ArrayList<>();
		proArgs.add("-args");
		proArgs.add(input("g"));
		proArgs.add(input("p"));
		proArgs.add(input("e"));
		proArgs.add(input("u"));
		proArgs.add(Double.toString(alpha));
		proArgs.add(Integer.toString(maxiter));
		proArgs.add(output("p"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
			
		fullDMLScriptName = getScript();

		double[][] g = readTwitter2010LinkMatrix(twitter2010EdgeList(), maxNodes, maxLines);
		int rows = g.length;
		double[][] p = getRandomMatrix(rows, 1, 1, 1, 1, -1);
		double[][] e = getRandomMatrix(rows, 1, 1, 1, 1, -1);
		double[][] u = getRandomMatrix(1, rows, 1, 1, 1, -1);
		writeInputMatrixWithMTD("g", g, true);
		writeInputMatrixWithMTD("p", p, true);
		writeInputMatrixWithMTD("e", e, true);
		writeInputMatrixWithMTD("u", u, true);

		for (int i = 0; i < maxiter; i++) {
			double[][] gp = TestUtils.performMatrixMultiplication(g, p);
			double[][] eu = TestUtils.performMatrixMultiplication(e, u);
			double[][] eup = TestUtils.performMatrixMultiplication(eu, p);
			for (int j = 0; j < rows; j++) {
				p[j][0] = alpha * gp[j][0] + (1 - alpha) * eup[j][0];
			}
		}

		writeExpectedMatrix("p", p);

		/*
		* No Spark job limit for the local edge list.
		*/
		int expectedNumberOfJobs = -1;
		runTest(true, EXCEPTION_NOT_EXPECTED, null, expectedNumberOfJobs);

		compareResults(0.0000001);
	}

	private static File twitter2010EdgeList() {
		String configured = System.getProperty("sysds.test.twitter2010");
		if (configured == null || configured.isEmpty())
			configured = System.getenv("TWITTER2010_EDGES");
		Assert.assertTrue(
			"Set -Dsysds.test.twitter2010 or TWITTER2010_EDGES to a local Twitter2010 edge list.",
			configured != null && !configured.isEmpty());
		File edgeList = new File(configured);
		Assert.assertTrue("Twitter2010 edge list not found: " + edgeList.getAbsolutePath(), edgeList.isFile());
		return edgeList;
	}

	// src -> dst stored as 1/outdegree in column src
	private static double[][] readTwitter2010LinkMatrix(File edgeList, int maxNodes, int maxLines) throws IOException {
		Map<Long, Integer> ids = new HashMap<>();
		List<int[]> edges = new ArrayList<>();
		Set<Long> seen = new HashSet<>();
		int lines = 0;
		try (BufferedReader reader = new BufferedReader(new FileReader(edgeList))) {
			String line;
			while ((line = reader.readLine()) != null && lines < maxLines) {
				line = line.trim();
				if (line.isEmpty() || line.charAt(0) == '#')
					continue;
				lines++;
				String[] parts = line.split("\\s+");
				if (parts.length < 2)
					continue;
				int src = internNode(ids, Long.parseLong(parts[0]), maxNodes);
				int dst = internNode(ids, Long.parseLong(parts[1]), maxNodes);
				if (src < 0 || dst < 0)
					continue;
				long key = (((long) src) << 32) | (dst & 0xffffffffL);
				if (seen.add(key))
					edges.add(new int[] {src, dst});
			}
		}

		int n = ids.size();
		Assert.assertTrue("No Twitter2010 edges were read from " + edgeList.getAbsolutePath(), n > 0 && !edges.isEmpty());
		int[] outDegree = new int[n];
		for (int[] edge : edges) {
			outDegree[edge[0]]++;
		}

		double[][] link = new double[n][n];
		for (int[] edge : edges) {
			link[edge[1]][edge[0]] = 1.0 / outDegree[edge[0]];
		}
		return link;
	}

	private static int internNode(Map<Long, Integer> ids, long node, int maxNodes) {
		Integer index = ids.get(node);
		if (index != null)
			return index;
		if (ids.size() >= maxNodes)
			return -1;
		int assigned = ids.size();
		ids.put(node, assigned);
		return assigned;
	}
}
