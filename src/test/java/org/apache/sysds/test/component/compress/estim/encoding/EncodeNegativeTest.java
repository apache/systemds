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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class EncodeNegativeTest {

	final MatrixBlock mock;

	public EncodeNegativeTest() {
		mock = new MatrixBlock(3, 3, new DenseBlockFP64Mock(new int[] {3, 3}, new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9}));
		mock.setNonZeros(9);
	}

	@Test(expected = NotImplementedException.class)
	public void encodeNonContiguous() {
		EncodingFactory.createFromMatrixBlock(mock, false, 3);
	}

	@Test(expected = NotImplementedException.class)
	public void encodeNonContiguousTransposed() {
		EncodingFactory.createFromMatrixBlock(mock, true, 3);
	}

	@Test(expected = NullPointerException.class)
	public void testInvalidToCallWithNullDeltaTransposed() {
		EncodingFactory.createFromMatrixBlockDelta(null, true, null);
	}

	@Test(expected = NullPointerException.class)
	public void testInvalidToCallWithNullDelta() {
		EncodingFactory.createFromMatrixBlockDelta(null, false, null);
	}

	@Test(expected = NullPointerException.class)
	public void testInvalidToCallWithNull() {
		EncodingFactory.createFromMatrixBlock(null, false, null);
	}

	@Test(expected = NotImplementedException.class)
	public void testDeltaTransposed() {
		MatrixBlock mb = new MatrixBlock(10, 10, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 1);
		mb.set(0, 1, 2);
		mb.setNonZeros(2);
		EncodingFactory.createFromMatrixBlockDelta(mb, true, ColIndexFactory.create(2));
	}

	@Test(expected = NullPointerException.class)
	public void testDelta() {
		EncodingFactory.createFromMatrixBlockDelta(new MatrixBlock(10, 10, false), false, null);
	}

	@Test(expected = NotImplementedException.class)
	public void testDeltaTransposedNVals() {
		MatrixBlock mb = new MatrixBlock(10, 10, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 1);
		mb.set(0, 1, 2);
		mb.setNonZeros(2);
		EncodingFactory.createFromMatrixBlockDelta(mb, true, ColIndexFactory.create(2), 2);
	}

	@Test(expected = NullPointerException.class)
	public void testDeltaNVals() {
		EncodingFactory.createFromMatrixBlockDelta(new MatrixBlock(10, 10, false), false, null, 1);
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
