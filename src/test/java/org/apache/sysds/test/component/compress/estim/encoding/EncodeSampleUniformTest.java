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

import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class EncodeSampleUniformTest extends EncodeSampleMultiColTest {

	public EncodeSampleUniformTest(MatrixBlock m, boolean t, int u, IEncode e, IEncode fh, IEncode sh) {
		super(m, t, u, e, fh, sh);
	}

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		// row reading dense
		tests.add(create(2, 30, 1.0, true, 2, 1251));
		tests.add(create(3, 30, 1.0, true, 2, 13142));
		tests.add(create(10, 30, 1.0, true, 2, 182828));

		// col reading dense
		tests.add(create(30, 2, 1.0, false, 2, 8865));
		tests.add(create(30, 3, 1.0, false, 2, 9876));
		tests.add(create(30, 10, 1.0, false, 2, 7654));

		// row sparse
		for(int i = 0; i < 5; i++) {
			tests.add(create(2, 300, 0.1, true, 2 , 1251 * i));
			tests.add(create(2, 300, 0.1, true, 2 , 11 * i));
			tests.add(create(2, 300, 0.2, true, 2 , 65 * i));
			tests.add(create(2, 300, 0.24, true, 2 , 245 * i));
			tests.add(create(2, 300, 0.24, true, 3 , 16 * i));
			tests.add(create(2, 300, 0.23, true, 3 , 15 * i));
		}

		// ultra sparse
		tests.add(create(2, 10000, 0.001, true, 3, 215));
		tests.add(create(2, 100000, 0.0001, true, 3, 42152));

		// const
		tests.add(create(3, 30, 1.0, true, 1, 2));
		tests.add(create(50, 5, 1.0, false, 1, 2));

		// empty
		tests.add(create(10, 10, 0.0, false, 1, 2));
		tests.add(create(100, 100, 0.0, false, 1, 2));

		return tests;
	}

	private static Object[] create(int nRow, int nCol, double sparsity, boolean t, int nUnique, int seed) {
		try {
			// Make sure that nUnique always is correct if we have a large enough matrix.
			nUnique -= sparsity < 1.0 ? 1 : 0;
			final int min = sparsity < 1.0 ? 0 : 1;

			MatrixBlock m = sparsity == 0.0 ? new MatrixBlock(nRow, nCol, true) : TestUtils
				.round(TestUtils.generateTestMatrixBlock(nRow, nCol, min, nUnique, sparsity, seed));

			return create(m, t);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed to initialize the Encoding test");
			return null; // this is never executed but java require it.
		}
	}

	public static Object[] create(MatrixBlock m, boolean t) {
		try {
			// Make sure that nUnique always is correct if we have a large enough matrix.

			final int d = t ? m.getNumRows() : m.getNumColumns();
			final IEncode e = EncodingFactory.createFromMatrixBlock(m, t, genRowCol(d));

			// split and read subparts individually
			final int dfh = d / 2;
			final IEncode fh = EncodingFactory.createFromMatrixBlock(m, t, genRowCol(dfh));
			final IEncode sh = EncodingFactory.createFromMatrixBlock(m, t, genRowCol(dfh, d));

			// join subparts and use its unique count for tests
			final IEncode er = fh.combine(sh);
			int u = er.getUnique();

			return new Object[] {m, t, u, e, fh, sh};
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed to initialize the Encoding test");
			return null; // this is never executed but java require it.
		}
	}
}
