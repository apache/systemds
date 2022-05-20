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

package org.apache.sysds.test.component.compress.colgroup;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.component.compress.CompressibleInputGenerator;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class JolEstimateRLETest extends JolEstimateTest {

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		MatrixBlock mb;

		// The size of the compression is the same even at different numbers of repeated values.
		tests.add(conv(new double[][] {{1}}));
		tests.add(conv(new double[][] {{0, 0, 0, 0, 5, 0}}));
		tests.add(conv(new double[][] {{0, 0, 0, 0, 5, 5, 0}}));
		tests.add(conv(new double[][] {{0, 0, 0, 0, 5, 5, 5, 0}}));
		tests.add(conv(new double[][] {{0, 0, 0, 0, 5, 5, 5, 5, 5, 5}}));
		tests.add(conv(new double[][] {{0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0}}));
		tests.add(conv(new double[][] {{0, 0, 0, 0, 5, 0, 5, 5, 5, 0, 5}}));
		tests.add(conv(new double[][] {{0, 5, 0, 5, 5, 5, 0, 5}}));
		tests.add(conv(new double[][] {{0, 5, 0, 5, 0, 5}}));
		tests.add(conv(new double[][] {{1, 5, 1, 5, 1, 5}}));
		tests.add(conv(new double[][] {{1, 1, 1, 5, 1, 5}}));
		tests.add(conv(new double[][] {{1, 1, 1, 5, 5, 5}}));
		tests.add(conv(new double[][] {{1, 1, 1, 5, 5, 1}}));

		// Worst case all random numbers dense.
		tests.add(gen(1, 100, 0, 100, 1.0, 7));
		tests.add(gen(1, 1000, 0, 100, 1.0, 7));
		tests.add(gen(1, 10000, 0, 100, 1.0, 7));

		// Random rounded numbers dense
		for(int seed = 0; seed < 5; seed++) {
			tests.add(genR(1, 30, 1, 4, 1.0, seed));
			tests.add(genR(1, 63, 1, 3, 1.0, seed));
			tests.add(genR(1, 64, 1, 3, 1.0, seed));
			tests.add(genR(1, 100, 1, 3, 1.0, seed));
			tests.add(genR(1, 128, 1, 3, 1.0, seed));
			tests.add(genR(1, 100, 1, 10, 1.0, seed));
		}
		tests.add(genR(1, 1523, 1, 99, 1.0, 7));
		tests.add(genR(1, 1523, 1, 3, 1.0, 7));
		tests.add(genR(1, 2000, 1, 3, 1.0, 7));
		tests.add(genR(1, 4000, 1, 255, 1.0, 7));

		// // Sparse rounded numbers
		// // Scale directly with sparsity
		tests.add(genR(1, 1523, 0, 99, 0.1, 7));
		tests.add(genR(1, 1621, 0, 99, 0.1, 142));
		tests.add(genR(1, 2321, 0, 99, 0.1, 512));
		tests.add(genR(1, 4000, 0, 255, 0.1, 7));

		// // Medium sparsity
		tests.add(genR(1, 1523, 0, 99, 0.5, 7));
		tests.add(genR(1, 1621, 0, 99, 0.5, 142));
		tests.add(genR(1, 2321, 0, 99, 0.5, 512));
		tests.add(genR(1, 4000, 0, 255, 0.5, 7));

		// Dream inputs 1 unique value
		tests.add(new Object[] {genRLE(10000, 1, 1, 2)});

		// when the rows length is larger than overflowing the character value,
		// the run gets split into two
		// char overflows into the next position increasing size by 1 char.
		int charMax = Character.MAX_VALUE;
		mb = genRLE(charMax, 1, 1, 132);
		tests.add(new Object[] {mb});
		mb = genRLE(charMax + 1, 1, 1, 132);
		tests.add(new Object[] {mb});
		mb = genRLE(charMax * 2 + 1, 1, 1, 132);
		tests.add(new Object[] {mb});

		// full run before start
		MatrixBlock mmb = new MatrixBlock(1, charMax, false);
		MatrixBlock rrmb = genRM(1, 100, 1, 3, 1.0, 4);
		tests.add(new Object[] {mmb.append(rrmb)});
		tests.add(new Object[] {mmb.append(rrmb).append(mmb)});
		tests.add(new Object[] {mmb.append(rrmb).append(mmb).append(rrmb)});
		tests.add(new Object[] {mmb.append(rrmb).append(mmb).append(rrmb).append(mmb)});
		tests.add(new Object[] {rrmb.append(mmb).append(rrmb)});

		// 10 unique values ordered such that all 10 instances is in the same run.
		// Results in same size no matter the number of original rows.

		mb = genRLE(100, 1, 10, 1);
		tests.add(new Object[] {mb});
		// mb = genRLE(1, 100, 10, 1);
		// LOG.error(mb);
		// tests.add(new Object[] {mb});
		mb = genRLE(1000, 1, 10, 1312);
		tests.add(new Object[] {mb});
		mb = genRLE(10000, 1, 10, 14512);
		tests.add(new Object[] {mb});
		mb = genRLE(100000, 1, 10, 132);
		tests.add(new Object[] {mb});

		// Sparse Dream inputs.
		mb = genRLE(100, 1, 10, 1);
		tests.add(new Object[] {mb});
		mb = genRLE(1000, 1, 10, 1312);
		tests.add(new Object[] {mb});
		mb = genRLE(10000, 1, 10, 14512);
		tests.add(new Object[] {mb});
		mb = genRLE(100000, 1, 10, 132);
		tests.add(new Object[] {mb});
		mb = genRLE(1000000, 1, 10, 132);
		tests.add(new Object[] {mb});
		mb = genRLE(1000000, 1, 1, 132);
		tests.add(new Object[] {mb});

		// Multi Column
		// two identical columns
		tests.add(new Object[] {genRLE(5, 1000, 2, 132)});
		// tests.add(new Object[] {genRLE(10, 2, 2, 132)});
		// tests.add(new Object[] {genRLE(10, 6, 2, 132)});
		// tests.add(new Object[] {genRLE(10, 100, 2, 132)});
		// tests.add(new Object[] {genRLE(101, 17, 2, 132)});
		// tests.add(new Object[] {genRLE(101, 17, 3, 132)});

		return tests;
	}

	public JolEstimateRLETest(MatrixBlock mb) {
		super(mb);
	}

	private static MatrixBlock genRLE(int row, int col, int runs, int seed) {
		return CompressibleInputGenerator.getInput(row, col, CompressionType.RLE, runs, 1.0, seed, true);
	}

	@Override
	public CompressionType getCT() {
		return rle;
	}
}
