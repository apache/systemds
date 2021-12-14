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

package org.apache.sysds.test.component.compress.estim.encoding;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;

import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class EncodeSampleDenseNonUniform extends EncodeSampleMultiColTest {

	public EncodeSampleDenseNonUniform(MatrixBlock m, boolean t, int u, IEncode e, IEncode fh, IEncode sh) {
		super(m, t, u, e, fh, sh);
	}

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		// allocate two matrices of either row or col size always dense.
		tests.add(create(10, 10, 0.1, 231, false));
		tests.add(create(100, 10, 0.1, 231, false));
		tests.add(create(10, 100, 0.1, 231, false));

		tests.add(create(10, 100, 0.1, 231, true));
		tests.add(create(100, 10, 0.1, 231, true));

		tests.add(create(10,10, 1.0, 12341, true));
		tests.add(create(10,10, 0.1, 12341, true));
		tests.add(create(10, 10, 0.7, 12341, true));

		tests.add(create(10, 10, 0.9, 3, true));
		tests.add(create(4, 10, 0.9, 32, true));
		tests.add(create(4, 10, 0.9, 4215, true));

		return tests;
	}

	private static Object[] create(int nRow, int nCol, double likelihoodEntireRowIsEmpty, int seed, boolean t) {

		MatrixBlock m = new MatrixBlock(nRow, nCol, false);
		m.allocateBlock();

		Random r = new Random(seed);
		double[] mV = m.getDenseBlockValues();
		int nnz = 0;

		for(int i = 0; i < nRow; i++) {
			if(r.nextDouble() > likelihoodEntireRowIsEmpty) {
				for(int j = i * nCol; j < i * nCol + nCol; j++)
					mV[j] = r.nextInt(3) + 1;

				nnz += nRow;
			}
		}

		m.setNonZeros(nnz * 2);
		return EncodeSampleUniformTest.create(m, t);
	}

}
