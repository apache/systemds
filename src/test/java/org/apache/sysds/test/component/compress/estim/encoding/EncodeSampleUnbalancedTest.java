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

import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class EncodeSampleUnbalancedTest extends EncodeSampleMultiColTest {

	public EncodeSampleUnbalancedTest(MatrixBlock m, boolean t, int u, IEncode e, IEncode fh, IEncode sh) {
		super(m, t, u, e, fh, sh);
	}

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		// one sparse one dense ... small size
		tests.add(createT(1, 1.0, 2, 1, 0.1, 2, 10, 326314));
		tests.add(createT(1, .1, 2, 1, 1.0, 2, 10, 512));

		// bigger making it sparse in one and dense in another
		tests.add(createT(1, .1, 2, 1, 1.0, 2, 100, 32141));
		tests.add(createT(1, 1.0, 2, 1, 0.1, 2, 100, 777));

		tests.add(createT(1, .1, 1, 1, 1.0, 1, 100, 32141));
		tests.add(createT(1, 1.0, 1, 1, 0.1, 1, 100, 777));

		tests.add(createT(1, .4, 1, 1, .4, 1, 100, 32141));
		tests.add(createT(1, .4, 1, 1, .4, 1, 100, 777));

		tests.add(createT(1, .4, 2, 1, .4, 2, 100, 32141));
		tests.add(createT(1, .4, 2, 1, .4, 2, 100, 777));

		tests.add(createT(1, .4, 3, 1, .4, 3, 100, 32141));
		tests.add(createT(1, .4, 3, 1, .4, 3, 100, 777));

		tests.add(createTSparse(1, .5, 3, 1, .5, 3, 100, 32141, true, true));
		tests.add(createTSparse(1, .5, 3, 1, .5, 3, 100, 777, true, true));
		tests.add(createTSparse(1, .2, 3, 1, 1.0, 3, 100, 3377, true, true));

		for(int i = 0; i < 10; i++) {

			tests.add(createTSparse(1, .01, 2, 1, .01, 2, 100, i * 231, true, true));
			tests.add(createTSparse(1, .1, 3, 1, .2, 3, 100, i * 231, true, true));
		}

		// big sparse
		tests.add(createT(1, 0.0001, 10, 1, 0.0000001, 2, 10000000, 1231));
		// more rows
		tests.add(createT(3, 0.0001, 10, 10, 0.0000001, 2, 10000000, 444));

		// Both Sparse and end dense joined
		tests.add(createT(1, 0.2, 10, 10, 0.1, 2, 1000, 1231521));


		tests.add(createT(1, 1.0, 100, 1, 1.0, 10,  10000, 132));
		tests.add(createT(1, 1.0, 1000, 1, 1.0, 10,  10000, 132));

		return tests;
	}

	private static Object[] createTSparse(int nRow1, double sp1, int nU1, int nRow2, double sp2, int nU2, int nCol,
		int seed, boolean forceSparse, boolean forceSparse2) {
		return create(nRow1, nCol, sp1, nU1, nRow2, nCol, sp2, nU2, seed, true, forceSparse, forceSparse2);
	}

	private static Object[] createT(int nRow1, double sp1, int nU1, int nRow2, double sp2, int nU2, int nCol, int seed) {
		return create(nRow1, nCol, sp1, nU1, nRow2, nCol, sp2, nU2, seed, true, false, false);
	}

	private static Object[] create(int nRow1, int nCol1, double sp1, int nU1, int nRow2, int nCol2, double sp2, int nU2,
		int seed, boolean t, boolean forceSparse, boolean forceSparse2) {
		try {
			// Make sure that nUnique always is correct if we have a large enough matrix.
			nU1 -= sp1 < 1.0 ? 1 : 0;
			final int min1 = sp1 < 1.0 ? 0 : 1;
			MatrixBlock m1 = TestUtils.round(TestUtils.generateTestMatrixBlock(nRow1, nCol1, min1, nU1, sp1, seed));
			nU2 -= sp2 < 1.0 ? 1 : 0;
			final int min2 = sp2 < 1.0 ? 0 : 1;
			MatrixBlock m2 = TestUtils
				.round(TestUtils.generateTestMatrixBlock(nRow2, nCol2, min2, nU2, sp2, seed * 21351));

			if(forceSparse)
				m1.denseToSparse(true);
			if(forceSparse2)
				m2.denseToSparse(true);

			return create(m1, m2, t);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed to initialize the Encoding test");
			return null; // this is never executed but java require it.
		}
	}

	protected static Object[] create(MatrixBlock m1, MatrixBlock m2, boolean t) {

		MatrixBlock m = m1.append(m2, null, !t);
		return create(m, m1, m2, t);
	}

	protected static Object[] create(MatrixBlock m, MatrixBlock m1, MatrixBlock m2, boolean t) {
		try {

			final IEncode e = EncodingFactory.createFromMatrixBlock(m, t,
				ColIndexFactory.create(t ? m.getNumRows() : m.getNumColumns()));

			// sub part.
			final IEncode fh = EncodingFactory.createFromMatrixBlock(m1, t,
				ColIndexFactory.create(t ? m1.getNumRows() : m1.getNumColumns()));
			final IEncode sh = EncodingFactory.createFromMatrixBlock(m2, t,
				ColIndexFactory.create(t ? m2.getNumRows() : m2.getNumColumns()));

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
