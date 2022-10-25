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

package org.apache.sysds.test.component.compress.colgroup.scheme;

import static org.junit.Assert.assertTrue;

import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class CLAEmptySchemeTest {

	private final AColGroup g;
	private final ICLAScheme sh;

	public CLAEmptySchemeTest() {
		g = new ColGroupEmpty(//
			new int[] {1, 3, 5} // Columns
		);
		sh = g.getCompressionScheme();
	}

	@Test
	public void testConstValid() {
		assertTrue(sh != null);
	}

	@Test
	public void testToSmallMatrix() {
		assertTrue(sh.encode(new MatrixBlock(1, 3, new double[] {//
			1.1, 1.2, 1.3})) == null);
	}

	@Test
	public void testWrongValuesSingleRow() {
		assertTrue(sh.encode(new MatrixBlock(1, 6, new double[] {//
			0.0, 1.1, 0.2, 1.2, 0.2, 1.2})) == null);
	}

	@Test
	public void testWrongValuesSingleRowV2() {
		assertTrue(sh.encode(new MatrixBlock(1, 6, new double[] {//
			0.0, 1.0, 0.2, 1.2, 0.2, 1.3})) == null);
	}

	@Test
	public void testValidEncodeSingleRow() {
		assertTrue(sh.encode(new MatrixBlock(1, 6, new double[] {//
			0.1, 0.0, 0.04, 0.0, 0.03, 0.0})) != null);
	}

	@Test
	public void testValidEncodeMultiRow() {
		assertTrue(sh.encode(new MatrixBlock(2, 6, new double[] {//
			132, 0.0, 241, 0.0, 142, 0.0, //
			132, 0.0, 241, 0.0, 142, 0.0, //
		})) != null);
	}

	@Test
	public void testValidEncodeMultiRowsLarger() {
		assertTrue(sh.encode(new MatrixBlock(2, 10, new double[] {//
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, //
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, //
		})) != null);
	}

	@Test
	public void testInvalidEncodeMultiRowsValue() {
		assertTrue(sh.encode(new MatrixBlock(4, 8, new double[] {//
			0.0, 0.0, 0.2, 0.0, 1.2, 0.0, 0.2, 1.3, //
			0.0, 0.0, 0.2, 0.0, 1.2, 0.0, 0.2, 1.3, //
			0.0, 0.0, 0.2, 0.0, 1.2, 0.0, 0.2, 1.3, //
			0.0, 0.0, 0.2, 0.0, 1.2, 0.0, 0.2, 1.3, //
		})) != null);
	}

	@Test
	public void testValidEncodeMultiRowDifferentValuesOtherColumns() {
		assertTrue(sh.encode(new MatrixBlock(4, 12, new double[] {//
			0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 0.4, 1.2, 0.3, 1.3, //
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 0.2, 1.2, 0.2, 1.3, //
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 0.2, 1.2, 0.1, 1.3, //
			0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 0.4, 1.2, 0.1, 1.3, //
		})) != null);
	}

	@Test
	public void testInvalidEncodeValueMultiRowMultiError() {
		assertTrue(sh.encode(new MatrixBlock(4, 6, new double[] {//
			0.0, 1.1, 0.2, 1.2, 0.2, 1.3, //
			0.0, 1.1, 0.2, 1.2, 0.2, 1.3, //
			0.0, 1.1, 0.2, 1.2, 0.2, 1.4, //
			0.0, 1.1, 0.2, 1.2, 0.2, 1.3, //
		})) == null);
	}

	@Test
	public void testInvalidEncodeMultiRow() {
		assertTrue(sh.encode(new MatrixBlock(4, 6, new double[] {//
			0.0, 1.3, 0.2, 1.2, 0.2, 1.3, //
			0.0, 1.1, 0.2, 1.2, 0.2, 1.3, //
			0.0, 1.1, 0.2, 1.2, 0.2, 1.4, //
			0.0, 1.1, 0.2, 1.2, 0.2, 1.3, //
		})) == null);
	}

	@Test
	public void testEncodeOtherColumns() {
		assertTrue(sh.encode(new MatrixBlock(4, 5, new double[] {//
			1.1, 0.2, 1.2, 0.2, 1.3, //
			1.1, 0.2, 1.2, 0.2, 1.3, //
			1.1, 0.2, 1.2, 0.2, 1.3, //
			1.1, 0.2, 1.2, 0.2, 1.3, //
		}), new int[] {0, 2, 4}// other columns
		) == null);
	}

	@Test
	public void testEncodeOtherColumnsValid() {
		assertTrue(sh.encode(new MatrixBlock(4, 8, new double[] {//
			0.0, 1.1, 0.0, 0.2, 0.0, 1.2, 0.2, 1.3, //
			0.0, 1.1, 0.0, 0.2, 0.0, 1.2, 0.2, 1.3, //
			0.0, 1.1, 0.0, 0.2, 0.0, 1.2, 0.2, 1.3, //
			0.0, 1.1, 0.0, 0.2, 0.0, 1.2, 0.2, 1.3, //
		}), new int[] {0, 2, 4}// other columns
		) != null);
	}

	@Test
	public void testEncodeOtherColumnsInvalid() {
		assertTrue(sh.encode(new MatrixBlock(4, 5, new double[] {//
			1.1, 0.2, 1.2, 0.2, 1.3, //
			1.1, 0.2, 1.2, 0.2, 1.3, //
			1.1, 0.2, 1.4, 0.2, 1.3, //
			1.1, 0.2, 1.2, 0.2, 1.3, //
		}), new int[] {0, 2, 4}// other columns
		) == null);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testInvalidArgument_1() {
		sh.encode(null, new int[] {0, 2, 4, 5});
	}

	@Test(expected = IllegalArgumentException.class)
	public void testInvalidArgument_2() {
		sh.encode(null, new int[] {0, 2});
	}

	@Test
	public void testSparse() {
		MatrixBlock mb = new MatrixBlock(4, 6, new double[] {//
			0.01, 0.0, 0.2, 0.0, 0.2, 0.0, //
			0.01, 0.0, 0.2, 0.0, 0.2, 0.0, //
			0.01, 0.0, 0.2, 0.0, 0.2, 0.0, //
			0.01, 0.0, 0.2, 0.0, 0.2, 0.0, //
		});

		MatrixBlock empty = new MatrixBlock(4, 1000, 0.0);
		mb = mb.append(empty);

		assertTrue(sh.encode(mb) != null);
	}

	@Test
	public void testSpars_AllCosOver() {
		MatrixBlock mb = new MatrixBlock(4, 6, new double[] {//
			0.01, 0.0, 0.2, 0.0, 0.2, 0.0, //
			0.01, 0.0, 0.2, 0.0, 0.2, 0.0, //
			0.01, 0.0, 0.2, 0.0, 0.2, 0.0, //
			0.01, 0.0, 0.2, 0.0, 0.2, 0.0, //
		});

		MatrixBlock empty = new MatrixBlock(4, 1000, 0.0);
		mb = mb.append(empty);

		assertTrue(sh.encode(mb, new int[] {100, 102, 999}) != null);
	}

	@Test
	public void testSpars_InsideInvalid() {
		MatrixBlock mb = new MatrixBlock(4, 6, new double[] {//
			0.01, 0.0, 0.2, 0.0, 0.2, 0.0, //
			0.01, 0.0, 0.2, 0.0, 0.2, 0.0, //
			0.01, 0.0, 0.2, 0.0, 0.2, 0.0, //
			0.01, 0.0, 0.2, 0.0, 0.2, 0.0, //
		});

		MatrixBlock empty = new MatrixBlock(4, 1000, 0.0);
		mb = mb.append(empty);

		assertTrue(sh.encode(mb, new int[] {1, 4, 5}) == null);
	}

	@Test
	public void testSparse_Append() {
		MatrixBlock mb = new MatrixBlock(4, 6, new double[] {//
			0.0, 0.0, 0.2, 0.0, 0.2, 1.3, //
			0.0, 0.0, 0.2, 0.0, 0.2, 1.3, //
			0.0, 0.0, 0.2, 0.0, 0.2, 1.3, //
			0.0, 0.0, 0.2, 0.0, 0.2, 1.3, //
		});

		MatrixBlock empty = new MatrixBlock(4, 1000, 0.0);
		mb = empty.append(mb);

		assertTrue(sh.encode(mb) != null);
	}

	@Test
	public void testSparseValidCustom() {
		MatrixBlock mb = new MatrixBlock(4, 9, new double[] {//
			0.0, 0.0, 1.1, 0.0, 0.2, 0.0, 1.2, 0.2, 1.3, //
			0.0, 0.0, 1.1, 0.0, 0.2, 0.0, 1.2, 0.2, 1.3, //
			0.0, 0.0, 1.1, 0.0, 0.2, 0.0, 1.2, 0.2, 1.3, //
			0.0, 0.0, 1.1, 0.0, 0.2, 0.0, 1.2, 0.2, 1.3, //
		});

		MatrixBlock empty = new MatrixBlock(4, 1000, 0.0);
		mb = empty.append(mb);

		assertTrue(sh.encode(mb, new int[] {1001, 1003, 1005}) != null);
	}

	@Test
	public void testSparseValidCustom2() {
		MatrixBlock mb = new MatrixBlock(4, 9, new double[] {//
			0.0, 0.0, 1.1, 0.0, 0.2, 0.0, 1.2, 0.2, 1.3, //
			0.0, 0.0, 1.1, 0.0, 0.2, 0.0, 1.2, 0.2, 1.3, //
			0.0, 0.0, 1.1, 0.0, 0.2, 0.0, 1.2, 0.2, 1.3, //
			0.0, 0.0, 1.1, 0.0, 0.2, 0.0, 1.2, 0.2, 1.3, //
		});

		MatrixBlock empty = new MatrixBlock(4, 1000, 0.0);
		MatrixBlock comb = empty.append(mb).append(mb);

		assertTrue(sh.encode(comb, new int[] {1001, 1003, 1005}) != null);
	}

	@Test
	public void testSparseValidCustom3Valid() {
		MatrixBlock mb = new MatrixBlock(4, 9, new double[] {//
			0.0, 0.0, 1.1, 0.0, 0.2, 0.0, 1.2, 0.2, 1.3, //
			0.0, 0.0, 1.1, 0.0, 0.2, 0.0, 1.33, 0.2, 1.3, //
			0.0, 0.0, 1.1, 0.0, 0.2, 0.0, 1.2, 0.2, 1.3, //
			0.0, 0.0, 1.1, 0.0, 0.2, 0.0, 1.2, 0.2, 1.3, //
		});

		MatrixBlock empty = new MatrixBlock(4, 1000, 0.0);
		MatrixBlock comb = empty.append(mb).append(mb);

		assertTrue(sh.encode(comb, new int[] {1001, 1003, 1005}) != null);
	}

	@Test
	public void testSparseEmptyRow() {
		MatrixBlock mb = new MatrixBlock(4, 6, new double[] {//
			0.0, 1.1, 0.2, 1.2, 0.2, 1.3, //
			0.0, 1.1, 0.2, 1.2, 0.2, 1.3, //
			0.0, 1.1, 0.2, 1.2, 0.2, 1.3, //
			0.0, 1.1, 0.2, 1.2, 0.2, 1.3, //
		});

		MatrixBlock empty = new MatrixBlock(4, 1000, 0.0);
		mb = empty.append(mb);
		MatrixBlock emptyRow = new MatrixBlock(1, 1006, 0.0);
		mb = mb.append(emptyRow, false);

		assertTrue(sh.encode(mb, new int[] {44, 45, 999}) != null);
	}

	@Test
	public void testEmpty() {
		MatrixBlock empty = new MatrixBlock(4, 1000, 0.0);
		assertTrue(sh.encode(empty) != null);
	}

	@Test
	public void testEmptyOtherColumns() {
		MatrixBlock empty = new MatrixBlock(4, 1000, 0.0);
		assertTrue(sh.encode(empty, new int[] {33, 34, 99}) != null);
	}

	@Test
	public void testGenericNonContinuosBlockValid() {
		MatrixBlock mb = new MatrixBlock(4, 6, //
			new DenseBlockFP64Mock(new int[] {4, 9}, new double[] {//
				0.2, 0.0, 1.1, 0.0, 0.4, 0.0, 1.2, 0.3, 1.3, //
				0.0, 0.0, 1.1, 0.0, 0.2, 0.0, 1.2, 0.2, 1.3, //
				0.0, 0.0, 1.1, 0.0, 0.2, 0.0, 1.2, 0.1, 1.3, //
				0.2, 0.0, 1.1, 0.0, 0.4, 0.0, 1.2, 0.1, 1.3, //
			}));
		mb.recomputeNonZeros();
		assertTrue(sh.encode(mb) != null);
	}

	@Test
	public void testGenericNonContinuosBlockInValid() {
		MatrixBlock mb = new MatrixBlock(4, 6, //
			new DenseBlockFP64Mock(new int[] {4, 6}, new double[] {//
				0.2, 1.1, 0.4, 1.2, 0.3, 1.3, //
				0.0, 1.1, 0.2, 1.2, 0.2, 1.3, //
				0.0, 1.1, 0.2, 1.2, 0.1, 1.3, //
				0.2, 1.22, 0.4, 1.2, 0.1, 1.3, //
			}));
		mb.recomputeNonZeros();
		assertTrue(sh.encode(mb) == null);
	}

	@Test(expected = NullPointerException.class)
	public void testNull() {
		sh.encode(null, null);
	}

	private class DenseBlockFP64Mock extends DenseBlockFP64 {
		private static final long serialVersionUID = -3601232958390554672L;

		public DenseBlockFP64Mock(int[] dims, double[] data) {
			super(dims, data);
		}

		@Override
		public boolean isContiguous() {
			return false;
		}

		@Override
		public int numBlocks() {
			return 2;
		}
	}

}
