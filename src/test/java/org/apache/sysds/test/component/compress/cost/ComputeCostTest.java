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

package org.apache.sysds.test.component.compress.cost;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import org.apache.sysds.runtime.compress.cost.ACostEstimate;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class ComputeCostTest extends ACostTest {

	public ComputeCostTest(MatrixBlock mb, ACostEstimate e, int seed) {
		super(mb, e, seed);
	}

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		Random r = new Random(1231);
		List<ACostEstimate> costEstimators = getComputeCosts();
		List<MatrixBlock> mbs = getMatrixBlocks(r);
		final int m = Integer.MAX_VALUE;
		for(MatrixBlock mb : mbs)
			for(ACostEstimate e : costEstimators)
				tests.add(new Object[] {mb, e, r.nextInt(m)});

		return tests;
	}

	private static List<ACostEstimate> getComputeCosts() {
		List<ACostEstimate> costEstimators = new ArrayList<>();
		// dictionary op that is densifying (plus)
		costEstimators.add(new ComputationCostEstimator(0, 0, 0, 0, 0, 0, 1, 0, true));
		// Left multiplication
		costEstimators.add(new ComputationCostEstimator(0, 0, 0, 1, 0, 0, 0, 0, false));
		// Left Multiplication but the matrix is densified
		costEstimators.add(new ComputationCostEstimator(0, 0, 0, 1, 0, 0, 1, 0, true));
		// 10 LMM densified
		costEstimators.add(new ComputationCostEstimator(0, 0, 0, 10, 0, 0, 1, 0, true));

		// Right Matrix Multiplication
		costEstimators.add(new ComputationCostEstimator(0, 0, 0, 0, 1, 0, 0, 0, false));
		costEstimators.add(new ComputationCostEstimator(0, 0, 0, 0, 1, 0, 1, 0, true));

		// Decompression
		costEstimators.add(new ComputationCostEstimator(0, 1, 0, 0, 0, 0, 0, 0, false));

		// decompressing after densifying
		costEstimators.add(new ComputationCostEstimator(0, 1, 0, 0, 0, 0, 1, 0, true));

		// One Scan (this is the type that is used if we
		// require a process through the index structure) such as in rowSum.
		costEstimators.add(new ComputationCostEstimator(1, 0, 0, 0, 0, 0, 0, 0, false));
		costEstimators.add(new ComputationCostEstimator(1, 0, 0, 0, 0, 0, 1, 0, true));

		// Overlapping decompression
		costEstimators.add(new ComputationCostEstimator(0, 0, 1, 0, 0, 0, 0, 0, false));

		// Compressed Multiplication
		costEstimators.add(new ComputationCostEstimator(0, 0, 0, 0, 0, 1, 0, 0, false));

		return costEstimators;
	}

	private static List<MatrixBlock> getMatrixBlocks(Random r) {
		List<MatrixBlock> mbs = new ArrayList<>();
		mbs.add(gen(1000, 2, 0, 5, 0.1, r));
		mbs.add(gen(3000, 10, 0, 5, 0.1, r));
		mbs.add(gen(3000, 10, 0, 5, 1.0, r));
		mbs.add(gen(3000, 10, 0, 0, 0, r)); // const empty
		mbs.add(gen(3000, 10, 1, 1, 1.0, r)); // const dense
		return mbs;
	}

	private static MatrixBlock gen(int nRow, int nCol, int min, int max, double s, Random r) {
		final int m = Integer.MAX_VALUE;
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(nRow, nCol, min, max, s, r.nextInt(m));
		return TestUtils.round(mb);
	}

}
