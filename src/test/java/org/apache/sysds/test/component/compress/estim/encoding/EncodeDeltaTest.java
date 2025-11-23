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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.EmptyEncoding;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class EncodeDeltaTest {

	protected static final Log LOG = LogFactory.getLog(EncodeDeltaTest.class.getName());

	@Test
	public void testCreateFromMatrixBlockDeltaBasic() {
		MatrixBlock mb = new MatrixBlock(3, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);
		mb.set(1, 0, 11);
		mb.set(1, 1, 21);
		mb.set(2, 0, 12);
		mb.set(2, 1, 22);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2));
		assertNotNull(encoding);
		assertTrue(encoding.getUnique() >= 1);
	}

	@Test
	public void testCreateFromMatrixBlockDeltaWithSampleSize() {
		MatrixBlock mb = new MatrixBlock(5, 2, false);
		mb.allocateDenseBlock();
		for(int i = 0; i < 5; i++) {
			mb.set(i, 0, 10 + i);
			mb.set(i, 1, 20 + i);
		}

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2), 3);
		assertNotNull(encoding);
	}

	@Test
	public void testCreateFromMatrixBlockDeltaFirstRowAsIs() {
		MatrixBlock mb = new MatrixBlock(2, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 5);
		mb.set(0, 1, 10);
		mb.set(1, 0, 5);
		mb.set(1, 1, 10);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2));
		assertNotNull(encoding);
	}

	@Test
	public void testCreateFromMatrixBlockDeltaConstantDeltas() {
		MatrixBlock mb = new MatrixBlock(4, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);
		mb.set(1, 0, 11);
		mb.set(1, 1, 21);
		mb.set(2, 0, 12);
		mb.set(2, 1, 22);
		mb.set(3, 0, 13);
		mb.set(3, 1, 23);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2));
		assertNotNull(encoding);
		assertTrue(encoding.getUnique() <= 2);
	}

	@Test
	public void testCreateFromMatrixBlockDeltaSingleRow() {
		MatrixBlock mb = new MatrixBlock(1, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2));
		assertNotNull(encoding);
	}

	@Test
	public void testCreateFromMatrixBlockDeltaSparse() {
		MatrixBlock mb = new MatrixBlock(3, 2, true);
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);
		mb.set(1, 0, 11);
		mb.set(2, 1, 22);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2));
		assertNotNull(encoding);
	}

	@Test
	public void testCreateFromMatrixBlockDeltaColumnSelection() {
		MatrixBlock mb = new MatrixBlock(3, 4, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);
		mb.set(0, 2, 30);
		mb.set(0, 3, 40);
		mb.set(1, 0, 11);
		mb.set(1, 1, 21);
		mb.set(1, 2, 31);
		mb.set(1, 3, 41);
		mb.set(2, 0, 12);
		mb.set(2, 1, 22);
		mb.set(2, 2, 32);
		mb.set(2, 3, 42);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(0, 2));
		assertNotNull(encoding);
	}

	@Test
	public void testCreateFromMatrixBlockDeltaNegativeValues() {
		MatrixBlock mb = new MatrixBlock(3, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);
		mb.set(1, 0, 8);
		mb.set(1, 1, 15);
		mb.set(2, 0, 12);
		mb.set(2, 1, 25);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2));
		assertNotNull(encoding);
	}

	@Test
	public void testCreateFromMatrixBlockDeltaZeros() {
		MatrixBlock mb = new MatrixBlock(3, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 5);
		mb.set(0, 1, 0);
		mb.set(1, 0, 5);
		mb.set(1, 1, 0);
		mb.set(2, 0, 0);
		mb.set(2, 1, 5);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2));
		assertNotNull(encoding);
	}

	@Test(expected = NotImplementedException.class)
	public void testCreateFromMatrixBlockDeltaTransposed() {
		MatrixBlock mb = new MatrixBlock(10, 10, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 1);
		mb.set(0, 1, 2);
		mb.setNonZeros(2);
		EncodingFactory.createFromMatrixBlockDelta(mb, true, ColIndexFactory.create(2));
	}

	@Test
	public void testCreateFromMatrixBlockDeltaLargeMatrix() {
		MatrixBlock mb = new MatrixBlock(100, 3, false);
		mb.allocateDenseBlock();
		for(int i = 0; i < 100; i++) {
			mb.set(i, 0, i);
			mb.set(i, 1, i * 2);
			mb.set(i, 2, i * 3);
		}

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(3));
		assertNotNull(encoding);
	}

	@Test
	public void testCreateFromMatrixBlockDeltaSampleSizeSmaller() {
		MatrixBlock mb = new MatrixBlock(10, 2, false);
		mb.allocateDenseBlock();
		for(int i = 0; i < 10; i++) {
			mb.set(i, 0, 10 + i);
			mb.set(i, 1, 20 + i);
		}

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2), 5);
		assertNotNull(encoding);
	}

	@Test
	public void testCreateFromMatrixBlockDeltaSampleSizeLarger() {
		MatrixBlock mb = new MatrixBlock(5, 2, false);
		mb.allocateDenseBlock();
		for(int i = 0; i < 5; i++) {
			mb.set(i, 0, 10 + i);
			mb.set(i, 1, 20 + i);
		}

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2), 10);
		assertNotNull(encoding);
	}

	@Test
	public void testCreateFromMatrixBlockDeltaEmptyMatrix() {
		// Test empty matrix with dimensions but all zeros
		MatrixBlock mb = new MatrixBlock(5, 2, false);
		mb.allocateDenseBlock();
		// Matrix has dimensions but is empty (all zeros)
		// isEmpty() should return true

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2));
		assertNotNull(encoding);
		assertTrue(encoding instanceof EmptyEncoding);
	}

	@Test
	public void testCreateFromMatrixBlockDeltaEmptyMatrixSparse() {
		// Test empty sparse matrix with dimensions
		MatrixBlock mb = new MatrixBlock(5, 2, true);
		// Sparse matrix with no values is empty
		mb.setNonZeros(0);

		IEncode encoding = EncodingFactory.createFromMatrixBlockDelta(mb, false, ColIndexFactory.create(2));
		assertNotNull(encoding);
		assertTrue(encoding instanceof EmptyEncoding);
	}

}

