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

package org.apache.sysds.test.component.compress.readers;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelection;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelectionDenseSingleBlockDelta;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelectionDenseMultiBlockDelta;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelectionSparseDelta;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelectionEmpty;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class ReadersDeltaTest {

	protected static final Log LOG = LogFactory.getLog(ReadersDeltaTest.class.getName());

	@Test
	public void testDeltaReaderDenseSingleBlockBasic() {
		MatrixBlock mb = new MatrixBlock(3, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);
		mb.set(1, 0, 11);
		mb.set(1, 1, 21);
		mb.set(2, 0, 12);
		mb.set(2, 1, 22);

		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(mb, ColIndexFactory.create(2), false);
		assertNotNull(reader);
		assertEquals(ReaderColumnSelectionDenseSingleBlockDelta.class, reader.getClass());

		DblArray row0 = reader.nextRow();
		assertNotNull(row0);
		assertArrayEquals(new double[] {10, 20}, row0.getData(), 0.0);

		DblArray row1 = reader.nextRow();
		assertNotNull(row1);
		assertArrayEquals(new double[] {1, 1}, row1.getData(), 0.0);

		DblArray row2 = reader.nextRow();
		assertNotNull(row2);
		assertArrayEquals(new double[] {1, 1}, row2.getData(), 0.0);

		assertNull(reader.nextRow());
	}

	@Test
	public void testDeltaReaderFirstRowAsIs() {
		MatrixBlock mb = new MatrixBlock(2, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 5);
		mb.set(0, 1, 10);
		mb.set(1, 0, 7);
		mb.set(1, 1, 12);

		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(mb, ColIndexFactory.create(2), false);
		DblArray row0 = reader.nextRow();
		assertArrayEquals(new double[] {5, 10}, row0.getData(), 0.0);
	}

	@Test
	public void testDeltaReaderNegativeValues() {
		MatrixBlock mb = new MatrixBlock(3, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);
		mb.set(1, 0, 8);
		mb.set(1, 1, 15);
		mb.set(2, 0, 12);
		mb.set(2, 1, 25);

		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(mb, ColIndexFactory.create(2), false);
		reader.nextRow();
		DblArray row1 = reader.nextRow();
		assertArrayEquals(new double[] {-2, -5}, row1.getData(), 0.0);

		DblArray row2 = reader.nextRow();
		assertArrayEquals(new double[] {4, 10}, row2.getData(), 0.0);
	}

	@Test
	public void testDeltaReaderZeros() {
		MatrixBlock mb = new MatrixBlock(3, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 5);
		mb.set(0, 1, 0);
		mb.set(1, 0, 5);
		mb.set(1, 1, 0);
		mb.set(2, 0, 0);
		mb.set(2, 1, 5);

		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(mb, ColIndexFactory.create(2), false);
		reader.nextRow();
		DblArray row1 = reader.nextRow();
		assertArrayEquals(new double[] {0, 0}, row1.getData(), 0.0);

		DblArray row2 = reader.nextRow();
		assertArrayEquals(new double[] {-5, 5}, row2.getData(), 0.0);
	}

	@Test
	public void testDeltaReaderSingleRow() {
		MatrixBlock mb = new MatrixBlock(1, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);

		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(mb, ColIndexFactory.create(2), false);
		DblArray row0 = reader.nextRow();
		assertNotNull(row0);
		assertArrayEquals(new double[] {10, 20}, row0.getData(), 0.0);
		assertNull(reader.nextRow());
	}

	@Test
	public void testDeltaReaderTwoRows() {
		MatrixBlock mb = new MatrixBlock(2, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);
		mb.set(1, 0, 15);
		mb.set(1, 1, 25);

		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(mb, ColIndexFactory.create(2), false);
		DblArray row0 = reader.nextRow();
		assertArrayEquals(new double[] {10, 20}, row0.getData(), 0.0);

		DblArray row1 = reader.nextRow();
		assertArrayEquals(new double[] {5, 5}, row1.getData(), 0.0);

		assertNull(reader.nextRow());
	}

	@Test
	public void testDeltaReaderColumnSelection() {
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

		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(mb, ColIndexFactory.createI(0, 2), false);
		DblArray row0 = reader.nextRow();
		assertArrayEquals(new double[] {10, 30}, row0.getData(), 0.0);

		DblArray row1 = reader.nextRow();
		assertArrayEquals(new double[] {1, 1}, row1.getData(), 0.0);

		DblArray row2 = reader.nextRow();
		assertArrayEquals(new double[] {1, 1}, row2.getData(), 0.0);
	}

	@Test
	public void testDeltaReaderSparse() {
		MatrixBlock mb = new MatrixBlock(3, 2, true);
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);
		mb.set(1, 0, 11);
		mb.set(2, 1, 22);

		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(mb, ColIndexFactory.create(2), false);
		assertNotNull(reader);
		assertEquals(ReaderColumnSelectionSparseDelta.class, reader.getClass());

		DblArray row0 = reader.nextRow();
		assertArrayEquals(new double[] {10, 20}, row0.getData(), 0.0);

		DblArray row1 = reader.nextRow();
		assertArrayEquals(new double[] {1, -20}, row1.getData(), 0.0);

		DblArray row2 = reader.nextRow();
		assertArrayEquals(new double[] {-11, 22}, row2.getData(), 0.0);
	}

	@Test
	public void testDeltaReaderSparseZeros() {
		MatrixBlock mb = new MatrixBlock(3, 2, true);
		mb.set(0, 0, 5);
		mb.set(1, 1, 10);
		mb.set(2, 0, 5);

		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(mb, ColIndexFactory.create(2), false);
		DblArray row0 = reader.nextRow();
		assertArrayEquals(new double[] {5, 0}, row0.getData(), 0.0);

		DblArray row1 = reader.nextRow();
		assertArrayEquals(new double[] {-5, 10}, row1.getData(), 0.0);

		DblArray row2 = reader.nextRow();
		assertArrayEquals(new double[] {5, -10}, row2.getData(), 0.0);
	}

	@Test
	public void testDeltaReaderRange() {
		MatrixBlock mb = new MatrixBlock(5, 2, false);
		mb.allocateDenseBlock();
		for(int i = 0; i < 5; i++) {
			mb.set(i, 0, 10 + i);
			mb.set(i, 1, 20 + i);
		}

		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(mb, ColIndexFactory.create(2), false, 1, 4);
		DblArray row1 = reader.nextRow();
		assertArrayEquals(new double[] {11, 21}, row1.getData(), 0.0);

		DblArray row2 = reader.nextRow();
		assertArrayEquals(new double[] {1, 1}, row2.getData(), 0.0);

		DblArray row3 = reader.nextRow();
		assertArrayEquals(new double[] {1, 1}, row3.getData(), 0.0);

		assertNull(reader.nextRow());
	}

	@Test(expected = DMLCompressionException.class)
	public void testDeltaReaderInvalidRange() {
		MatrixBlock mb = new MatrixBlock(10, 2, false);
		mb.allocateDenseBlock();
		ReaderColumnSelection.createDeltaReader(mb, ColIndexFactory.create(2), false, 10, 9);
	}

	@Test(expected = NotImplementedException.class)
	public void testDeltaReaderTransposed() {
		MatrixBlock mb = new MatrixBlock(10, 10, false);
		mb.allocateDenseBlock();
		mb.setNonZeros(100);
		ReaderColumnSelection.createDeltaReader(mb, ColIndexFactory.create(2), true);
	}

	@Test
	public void testDeltaReaderLargeMatrix() {
		MatrixBlock mb = new MatrixBlock(100, 3, false);
		mb.allocateDenseBlock();
		for(int i = 0; i < 100; i++) {
			mb.set(i, 0, i);
			mb.set(i, 1, i * 2);
			mb.set(i, 2, i * 3);
		}

		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(mb, ColIndexFactory.create(3), false);
		DblArray row0 = reader.nextRow();
		assertArrayEquals(new double[] {0, 0, 0}, row0.getData(), 0.0);

		for(int i = 1; i < 100; i++) {
			DblArray row = reader.nextRow();
			assertNotNull(row);
			assertArrayEquals(new double[] {1, 2, 3}, row.getData(), 0.0);
		}

		assertNull(reader.nextRow());
	}

	@Test
	public void testDeltaReaderEmptyMatrix() {
		// Test empty matrix with dimensions but all zeros
		MatrixBlock mb = new MatrixBlock(5, 2, false);
		mb.allocateDenseBlock();
		// Matrix has dimensions but is empty (all zeros)
		// isEmpty() should return true

		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(mb, ColIndexFactory.create(2), false);
		assertNotNull(reader);
		assertTrue(reader instanceof ReaderColumnSelectionEmpty);

		// Empty reader should return null immediately
		assertNull(reader.nextRow());
	}

	@Test
	public void testDeltaReaderEmptyMatrixSparse() {
		// Test empty sparse matrix with dimensions
		MatrixBlock mb = new MatrixBlock(5, 2, true);
		// Sparse matrix with no values is empty
		mb.setNonZeros(0);

		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(mb, ColIndexFactory.create(2), false);
		assertNotNull(reader);
		assertTrue(reader instanceof ReaderColumnSelectionEmpty);

		// Empty reader should return null immediately
		assertNull(reader.nextRow());
	}

}

