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

package org.apache.sysml.test.integration.mlcontext.algorithms;

import static org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromFile;

import java.util.concurrent.ThreadLocalRandom;

import org.apache.log4j.Logger;
import org.apache.sysml.api.mlcontext.Script;
import org.apache.sysml.test.integration.mlcontext.MLContextTestBase;
import org.junit.Assert;
import org.junit.Test;

public class MLContextUnivariateStatisticsTest extends MLContextTestBase {
	protected static Logger log = Logger.getLogger(MLContextUnivariateStatisticsTest.class);

	protected final static String TEST_SCRIPT = "scripts/algorithms/Univar-Stats.dml";

	@Test
	public void testRandomMatrix() {
		double[][] random10x3 = getRandomMatrix(10, 3, 0.0, 10.0, 0.9, -1);
		double[][] types = new double[][] { { 1.0, 1.0, 1.0 } };
		Script univarStats = dmlFromFile(TEST_SCRIPT);
		univarStats.in("A", random10x3).in("K", types).in("$CONSOLE_OUTPUT", true).out("baseStats");
		ml.execute(univarStats);
	}

	@Test
	public void testRandomMatrixWithRandomCategoricalColumn() {
		double[][] random10x3 = getRandomMatrix(10, 3, 0.0, 10.0, 0.9, -1);
		log.debug("Matrix before random int column replace:\n" + getMatrixAsString(random10x3));
		replaceColumnWithRandomInts(random10x3, 2, 1, 2);
		log.debug("Matrix after random int column replace:\n" + getMatrixAsString(random10x3));
		double[][] types = new double[][] { { 1.0, 1.0, 2.0 } };
		Script univarStats = dmlFromFile(TEST_SCRIPT);
		univarStats.in("A", random10x3).in("K", types).out("baseStats");
		ml.execute(univarStats);
	}

	@Test
	public void testScaleColumn() {
		double[][] matrix = new double[][] { { 1.0 }, { 2.0 }, { 2.0 }, { 3.0 }, { 4.0 } };
		double[][] types = new double[][] { { 1.0 } };
		Script univarStats = dmlFromFile(TEST_SCRIPT);
		univarStats.in("A", matrix).in("K", types).out("baseStats");
		double[][] stats = ml.execute(univarStats).getMatrix("baseStats").to2DDoubleArray();
		log.debug("Stats for scale column:\n" + getMatrixAsString(stats));
		Assert.assertEquals(1.0, stats[0][0], 0); // minimum
		Assert.assertEquals(4.0, stats[1][0], 0); // maximum
		Assert.assertEquals(2.4, stats[3][0], 0); // average
		Assert.assertEquals(2.0, stats[12][0], 0); // mean
	}

	@Test
	public void testCategoricalColumn() {
		double[][] matrix = new double[][] { { 1.0 }, { 2.0 }, { 2.0 }, { 3.0 }, { 4.0 } };
		double[][] types = new double[][] { { 2.0 } };
		Script univarStats = dmlFromFile(TEST_SCRIPT);
		univarStats.in("A", matrix).in("K", types).out("baseStats");
		double[][] stats = ml.execute(univarStats).getMatrix("baseStats").to2DDoubleArray();
		log.debug("Stats for categorical column:\n" + getMatrixAsString(stats));
		Assert.assertEquals(4.0, stats[14][0], 0); // number of categories
		Assert.assertEquals(2.0, stats[15][0], 0); // mode
		Assert.assertEquals(1.0, stats[16][0], 0); // number of modes
	}

	private static void replaceColumnWithRandomInts(double[][] matrix, int whichColumn, int lowValue, int highValue) {
		for (int i = 0; i < matrix.length; i++) {
			double[] row = matrix[i];
			row[whichColumn] = ThreadLocalRandom.current().nextInt(lowValue, highValue + 1);
		}
	}

}
