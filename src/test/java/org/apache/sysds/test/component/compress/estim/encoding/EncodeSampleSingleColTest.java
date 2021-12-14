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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class EncodeSampleSingleColTest extends EncodeSampleTest {

	protected static final Log LOG = LogFactory.getLog(EncodeSampleTest.class.getName());

	public EncodeSampleSingleColTest(MatrixBlock m, boolean t, int u, IEncode e) {
		super(m, t, u, e);
	}

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		
		tests.add(create(1, 10, 1.0, true, 2, 2));
		tests.add(create(1, 10, 0.5, true, 2, 131241));
		tests.add(create(1, 100, 1.0, true, 2, 1312));
		tests.add(create(1, 500, 0.5, true, 2, 132));
		tests.add(create(1, 500, 0.5, true, 10, 12));
		tests.add(create(10, 1, 1.0, false, 2, 32141));
		tests.add(create(10, 1, 0.5, false, 2, 132));
		tests.add(create(100, 1, 0.5, false, 2, 21));
		tests.add(create(100, 2, 0.5, false, 2, 3131));
		tests.add(create(100, 4, 0.5, false, 2, 32141));
		tests.add(create(100, 4, 0.5, false, 10, 11));
		
		tests.add(create(1, 100, 0.2, true, 2, 13));
		tests.add(create(1, 1000, 0.2, true, 10, 2));
		tests.add(create(1, 10000, 0.02, true, 10, 3145));
		tests.add(create(1, 100000, 0.002, true, 10, 3214));
		tests.add(create(1, 1000000, 0.0002, true, 10, 3232));
		
		tests.add(create(100, 100, 0.02, false, 2, 32));
		tests.add(create(1000, 100, 0.06, false, 2, 33412));
		
		// const
		tests.add(create(1, 10, 1.0, true, 1, 1341));
		tests.add(create(10, 1, 1.0, true, 1, 13));
		// tests.add(create(1, 10, 1.0, true, 1, 2));
		
		// empty
		tests.add(create(1, 10, 0.0, true, 1, 2));
		tests.add(create(10, 1, 0.0, false, 1, 2));

		tests.add(createEmptyAllocatedSparse(1, 10 ,true));
		tests.add(createEmptyAllocatedSparse(10, 1 ,false));
		
		return tests;
	}

	public static Object[] create(int nRow, int nCol, double sparsity, boolean transposed, int nUnique, int seed) {
		try {
			int u = nUnique;
			// Make sure that nUnique always is correct if we have a large enough matrix.
			nUnique -= sparsity < 1.0 ? 1 : 0;
			MatrixBlock m = TestUtils
				.round(TestUtils.generateTestMatrixBlock(nRow, nCol, sparsity < 1.0 ? 0 : 1, nUnique, sparsity, seed));

			boolean t = transposed;

			IEncode e = IEncode.createFromMatrixBlock(m, t, 0);
			return new Object[] {m, t, u, e};
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed to initialize the Encoding test");
			return null; // this is never executed but java require it.
		}
	}


	public static Object[] createEmptyAllocatedSparse(int nRow, int nCol, boolean transposed) {
		try {
			int u = 1;
			MatrixBlock m = new MatrixBlock(nRow, nCol, true);
			m.allocateBlock();
			m.setNonZeros(1);

			boolean t = transposed;

			IEncode e = IEncode.createFromMatrixBlock(m, t, 0);
			return new Object[] {m, t, u, e};
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed to initialize the Encoding test");
			return null; // this is never executed but java require it.
		}
	}
}
