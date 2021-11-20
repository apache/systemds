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

import org.apache.sysds.runtime.matrix.data.LibCommonsMath;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.Assert;

public class EigenDecompTest {
	@Test
	public void testEigenDecomp() {
		MatrixBlock in = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 1.0, 1);
		MatrixBlock[] m1 = LibCommonsMath.multiReturnOperations(in, "eigen");
		MatrixBlock[] m2 = LibCommonsMath.multiReturnOperations(in, "eigenours");
		if(m1 != null && m1.length == 2 && m1[0] != null && m1[1] != null &&
		   m2 != null && m2.length == 2 && m2[0] != null && m2[1] != null) {
			TestUtils.compareMatrices(m1[0], m2[0], 0.01, "Result of eigenvalues of new eigendecomp function wrong");
			TestUtils.compareMatrices(m1[1], m2[1], 0.01, "Result of eigenvectors of new eigendecomp function wrong");
		}
		else {
			Assert.fail("Wrong number of matrices returned from eigendecomp (or null)");
		}
	}

	@Test
	public void testTred2() {
		MatrixBlock in = new MatrixBlock(4, 4, false);
		double[] a = { 4, 1, -2,  2,
					   1, 2,  0,  1,
		              -2, 0,  3, -2,
				       2, 1, -2, -1};
		in.init(a, 4, 4);
		MatrixBlock[] m = LibCommonsMath.multiReturnOperations(in, "eigenours");
	}
}
