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

package org.apache.sysds.test.component.matrix;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.matrix.data.LibMatrixDatagen;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.RandomMatrixGenerator;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertArrayEquals;

public class LibMatrixDatagenTest {
	protected static final Log LOG = LogFactory.getLog(LibMatrixDatagenTest.class.getName());

	@Test
	public void testGenerateUniformMatrixPhilox() {
		MatrixBlock mb = new MatrixBlock();
		RandomMatrixGenerator rgen = new RandomMatrixGenerator(RandomMatrixGenerator.PDF.CB_UNIFORM, 10, 10, 10, 1, 0., 1.);
		LibMatrixDatagen.generateRandomMatrix(mb, rgen, null, 0L);
		for(int i = 0; i < 10; i++) {
			for(int j = 0; j < 10; j++) {
				assertTrue("Value: " + mb.get(i, j) + "needs to be less than 1", mb.get(i, j) < 1);
				assertTrue("Value: " + mb.get(i, j) + "needs to be greater than 0", mb.get(i, j) > 0);
			}
		}
	}

	@Test
	public void testGenerateNormalMatrixPhilox() {
		MatrixBlock mb = new MatrixBlock();
		RandomMatrixGenerator rgen = new RandomMatrixGenerator(RandomMatrixGenerator.PDF.CB_NORMAL, 1000, 1000, 1000 * 1000, 1);
		LibMatrixDatagen.generateRandomMatrix(mb, rgen, null, 123123123123L);
		double mean = mb.mean();
		double[] bv = mb.getDenseBlockValues();
		double variance = Arrays.stream(bv).map(x -> Math.pow(x - mean, 2)).sum() / bv.length;
		assertEquals("Mean should be 0", 0, mean, 0.01);
		assertEquals("Variance should be 1", 1, variance, 0.001);
	}

	@Test
	public void testGenerateUniformMatrixPhiloxShouldHaveGoodStatistics() {
		MatrixBlock mb = new MatrixBlock();
		RandomMatrixGenerator rgen = new RandomMatrixGenerator(RandomMatrixGenerator.PDF.CB_UNIFORM, 1000, 1000, 100, 1, 0., 1.);
		LibMatrixDatagen.generateRandomMatrix(mb, rgen, null, 0L);

		double mean = mb.mean();
		assertEquals("Mean should be 0.5", 0.5, mean, 0.001);

		double[] bv = mb.getDenseBlockValues();
		assertEquals(1000 * 1000, bv.length);
		double variance = Arrays.stream(bv).map(x -> Math.pow(x - mean, 2)).sum() / bv.length;
		assertEquals("Variance should be 1", 0.0833, variance, 0.001);
	}

	@Test
	public void testGenerateUniformMatrixShouldReturnSameValuesUsingStreams() {
		MatrixBlock mb = new MatrixBlock();
		RandomMatrixGenerator rgen = new RandomMatrixGenerator(RandomMatrixGenerator.PDF.UNIFORM, 1000, 1000, 100, 1, 0., 1.);
		LibMatrixDatagen.generateRandomMatrix(mb, rgen, null, 0L);

		double[] bv = Arrays.copyOf(mb.getDenseBlockValues(), 100);
		double[] previous = new double[] {0.24053641567148587, 0.6374174253501083, 0.5504370051176339, 0.5975452777972018, 0.3332183994766498, 0.3851891847407185, 0.984841540199809, 0.8791825178724801, 0.9412491794821144, 0.27495396603548483, 0.12889715087377673, 0.14660165764651822, 0.023238122483889456, 0.5467397571984656, 0.9644868606768501, 0.10449068625097169, 0.6251463634655593, 0.4107961954910617, 0.7763122912749325, 0.990722785714783, 0.4872328470301428, 0.7462414053223305, 0.7331520701949938, 0.8172970714093244, 0.8388903500470183, 0.5266994346048661, 0.8993350116114935, 0.13393984058689223, 0.0830623982249149, 0.9785743401478403, 0.7223571191888487, 0.7150310138504744, 0.14322038530059678, 0.4629578184224229, 0.004485602182885184, 0.07149831487989411, 0.34842022979166454, 0.3387696535357536, 0.859356551354648, 0.9715469888517128, 0.8657458802140383, 0.6125811047098682, 0.17898798452881726, 0.21757041220968598, 0.8544871670422907, 0.009673497300974332, 0.6922930069529333, 0.7713129661706796, 0.7126874281456893, 0.2112353749298962, 0.7830924897671794, 0.945333238959629, 0.014236355103667941, 0.3942035527773311, 0.8537907753080728, 0.7860424508145526, 0.993471955005814, 0.883104405981479, 0.17029153024770394, 0.9620689182075386, 0.7242950335788688, 0.6773541612498745, 0.8043954172246357, 0.44142677367579175, 0.46208799028599445, 0.8528274665994607, 0.501834850205735, 0.9919429804102169, 0.9692699099404161, 0.35310607217911816, 0.047265869196129406, 0.0716236234178006, 0.02910751272163581, 0.48367019010510015, 0.9719501209537452, 0.9891171507514055, 0.7674421030154899, 0.5013973510122299, 0.2555253108964435, 0.30915818724818767, 0.8482805002723425, 0.052084538173983286, 0.010175454536229256, 0.35385296970871194, 0.08673785516572752, 0.8503115152643057, 0.0036769023557003955, 0.3078931676344727, 0.5316085562487977, 0.9188142018385732, 0.27721002606871137, 0.8742622102831944, 0.6098815135127635, 0.9086392096967358, 0.04449062015679506, 0.6467239010388895, 0.4968037636226561, 0.5067015959528527, 0.5206888198929495, 0.36636074451399603};
		assertArrayEquals(previous, bv, 0.0001);
	}
}
