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

package org.apache.sysds.test.component.matrix.binary;

import static org.junit.Assert.assertEquals;

import org.apache.sysds.runtime.matrix.data.LibMatrixBincell;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell.BinaryAccessType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class GetAccessTypeTest {
	@Test
	public void getMatrix() {
		BinaryAccessType MM = BinaryAccessType.MATRIX_MATRIX;
		for(int i = 1; i < 100; i++) {
			assertEquals(MM, gat(m(i, i), m(i, i)));
		}
	}

	@Test
	public void getMcV() {
		BinaryAccessType MM = BinaryAccessType.MATRIX_COL_VECTOR;
		for(int i = 2; i < 100; i++) {
			assertEquals(MM, gat(m(i, i), m(i, 1)));
			assertEquals(String.format("%d,%d - %d,%d", i, i + 1, i, 1), MM, gat(m(i, i + 1), m(i, 1)));
		}
	}

	@Test
	public void getMrV() {
		BinaryAccessType MM = BinaryAccessType.MATRIX_ROW_VECTOR;
		for(int i = 2; i < 100; i++) {
			assertEquals(MM, gat(m(i, i), m(1, i)));
		}
	}

	@Test
	public void getVoV() {
		BinaryAccessType MM = BinaryAccessType.OUTER_VECTOR_VECTOR;
		for(int i = 2; i < 100; i++) {
			assertEquals(MM, gat(m(i, 1), m(1, i)));
		}
	}

	@Test
	public void getInvalid() {
		BinaryAccessType MM = BinaryAccessType.INVALID;
		for(int i = 2; i < 100; i++) {
			assertEquals(MM, gat(m(2, i + 1), m(1, i + 2)));
			assertEquals(MM, gat(m(2, i + 1), m(2, i + 2)));
			assertEquals(MM, gat(m(i, i + 1), m(i, i + 2)));
			assertEquals(MM, gat(m(i + 1, i), m(i + 2, i)));
			assertEquals(MM, gat(m(i, i + 1), m(i, i + 2)));
			assertEquals(MM, gat(m(i, 1), m(i, i + 2)));
			assertEquals(MM, gat(m(i, i), m(1, i + 2)));

		}
	}

	private MatrixBlock m(int r, int c) {
		return new MatrixBlock(r, c, true);
	}

	private BinaryAccessType gat(MatrixBlock a, MatrixBlock b) {
		return LibMatrixBincell.getBinaryAccessType(a, b);
	}
}
