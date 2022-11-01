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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.Iterator;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.iterators.IteratorFactory;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Assert;
import org.junit.Test;

public class EntityResolutionClusteringTest extends AutomatedTestBase {
	private final static String TEST_NAME = "EntityResolutionClustering";
	private final static String TEST_DIR = "applications/entity_resolution/clustering/";

	enum BlockingMethod {
		NAIVE,
		LSH,
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_DIR, TEST_NAME);
	}

	@Test
	public void testNaive1() throws IOException {
		testScriptEndToEnd(0.3, 1, BlockingMethod.NAIVE, 0, 0);
	}
	@Test
	public void testLSH1() throws IOException {
		testScriptEndToEnd(0.3, 1, BlockingMethod.LSH, 1, 1);
	}
	@Test
	public void testLSH2() throws IOException {
		testScriptEndToEnd(0.3, 1, BlockingMethod.LSH, 1, 3);
	}
	@Test
	public void testLSH3() throws IOException {
		testScriptEndToEnd(0.3, 1, BlockingMethod.LSH, 3, 1);
	}
	@Test
	public void testLSH4() throws IOException {
		testScriptEndToEnd(0.3, 1, BlockingMethod.LSH, 5, 5);
	}

	public void testScriptEndToEnd(double threshold, int numBlocks, BlockingMethod blockingMethod, int numLshHashtables, int numLshHyperplanes) throws IOException {
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);
		fullDMLScriptName = "./scripts/staging/entity-resolution/entity-clustering.dml";

		programArgs = new String[]{
				"-nvargs", //
				"FX=" + sourceDirectory + "input.csv", //
				"OUT=" + output("B"), //
				"threshold=" + threshold,
				"num_blocks=" + numBlocks,
				"blocking_method=" + (blockingMethod == BlockingMethod.LSH ? "lsh" : "naive"),
				"num_hashtables=" + numLshHashtables,
				"num_hyperplanes=" + numLshHyperplanes,
		};

		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

		// LSH is not deterministic, so in this test we just assert that it runs and produces a file
		if (blockingMethod == BlockingMethod.LSH) {
			Assert.assertTrue(Files.exists(Paths.get(output("B"))));
			return;
		}

		Files.copy(Paths.get(sourceDirectory + "expected.csv"), Paths.get(output("expected.csv")), StandardCopyOption.REPLACE_EXISTING);
		Files.copy(Paths.get(sourceDirectory + "expected.csv.mtd"), Paths.get(output("expected.csv.mtd")), StandardCopyOption.REPLACE_EXISTING);

		FrameBlock expectedPairs = readDMLFrameFromHDFS("expected.csv", Types.FileFormat.CSV);
		FrameBlock predictedPairs = readDMLFrameFromHDFS("B", Types.FileFormat.CSV);


		Iterator<Object[]> expectedIter = IteratorFactory.getObjectRowIterator(expectedPairs);
		Iterator<Object[]> predictedIter = IteratorFactory.getObjectRowIterator(predictedPairs);

		int row = 0;
		while (expectedIter.hasNext()) {
			Assert.assertTrue(predictedIter.hasNext());
			Object[] expected = Arrays.copyOfRange(expectedIter.next(), 0, 2);
			Object[] predicted = Arrays.copyOfRange(predictedIter.next(), 0, 2);
			Assert.assertArrayEquals("Row " + row + " differs.", expected, predicted);
			row++;
		}
		Assert.assertEquals(expectedPairs.getNumRows(), predictedPairs.getNumRows());
		tearDown();
	}
}
