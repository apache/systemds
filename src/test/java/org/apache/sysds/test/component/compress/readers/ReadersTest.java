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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelection;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.compress.utils.DblArrayCountHashMap;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Ignore;
import org.junit.Test;

public class ReadersTest {

	protected static final Log LOG = LogFactory.getLog(ReadersTest.class.getName());

	@Test(expected = DMLCompressionException.class)
	public void testDenseSingleCol() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(10, 1, 1, 1, 0.5, 21342);
		ReaderColumnSelection.createReader(mb,ColIndexFactory.create( 1), false);
	}

	@Test
	public void testDenseMultiCol() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(10, 1, 1, 1, 0.5, 21342);
		MatrixBlock mbc = mb.append(mb, null);
		ReaderColumnSelection r = ReaderColumnSelection.createReader(mbc, ColIndexFactory.create(2), false);
		DblArray d = null;
		DblArrayCountHashMap map = new DblArrayCountHashMap(4, 2);

		int i = 0;
		while((d = r.nextRow()) != null) {
			map.increment(d);
			i++;
		}

		assertEquals(mb.getNonZeros(), i);
	}

	@Test
	public void testDenseMultiColSparseRight() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(10, 1, 1, 1, 0.5, 21342);
		MatrixBlock mbc = mb.append(new MatrixBlock(10, 1, true), null);
		ReaderColumnSelection r = ReaderColumnSelection.createReader(mbc, ColIndexFactory.create(2), false);
		DblArray d = null;
		DblArrayCountHashMap map = new DblArrayCountHashMap(4, 2);

		int i = 0;
		while((d = r.nextRow()) != null) {
			map.increment(d);
			i++;
		}
		assertEquals(mb.getNonZeros(), i);
	}

	@Test
	public void testDenseMultiColSparseLeft() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(10, 1, 1, 1, 0.5, 21342);
		MatrixBlock mbc = new MatrixBlock(10, 1, true).append(mb, null);
		ReaderColumnSelection r = ReaderColumnSelection.createReader(mbc, ColIndexFactory.create(2), false);
		DblArray d = null;
		DblArrayCountHashMap map = new DblArrayCountHashMap(4, 2);

		int i = 0;
		while((d = r.nextRow()) != null) {
			map.increment(d);
			i++;
		}

		assertEquals(mb.getNonZeros(), i);
	}

	@Test
	public void testSpecificMultiCol() {

		// 4.0 0.0
		// 3.0 0.0
		// 0.0 5.0

		MatrixBlock mb = new MatrixBlock(3, 2, false);
		mb.allocateDenseBlock();
		mb.setValue(0, 0, 4);
		mb.setValue(1, 0, 3);
		mb.setValue(2, 1, 5);

		ReaderColumnSelection r = ReaderColumnSelection.createReader(mb, ColIndexFactory.create(2), false);
		DblArray d = null;
		DblArrayCountHashMap map = new DblArrayCountHashMap(4, 2);

		int i = 0;
		while((d = r.nextRow()) != null) {
			map.increment(d);
			i++;
		}
		assertEquals(i, 3);
	}

	@Test(expected = DMLCompressionException.class)
	public void testEmpty() {
		ReaderColumnSelection.createReader(new MatrixBlock(), ColIndexFactory.create(2), false);
	}

	@Test(expected = DMLCompressionException.class)
	public void testInvalidRange() {
		ReaderColumnSelection.createReader(new MatrixBlock(), ColIndexFactory.create(2), false, 10, 9);
	}

	@Test(expected = DMLCompressionException.class)
	public void testInvalidRange_02() {
		MatrixBlock mb = new MatrixBlock(10, 32, true);
		mb.allocateDenseBlock();
		ReaderColumnSelection.createReader(mb, ColIndexFactory.create(2), false, 10, 9);
	}

	@Test
	public void isEmptyNan() {
		try {

			MatrixBlock mb = new MatrixBlock(10, 5, Double.NaN);

			ReaderColumnSelection reader = ReaderColumnSelection.createReader(mb, ColIndexFactory.create(2), false, 0,
				mb.getNumRows());
			assertEquals(null, reader.nextRow());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void isNaN() {
		try {

			MatrixBlock mb = new MatrixBlock(10, 5, Double.NaN);
			mb.setValue(1, 1, 3214);

			ReaderColumnSelection reader = ReaderColumnSelection.createReader(mb, ColIndexFactory.create(2), false, 0,
				mb.getNumRows());
			DblArray a = reader.nextRow();
			assertNotEquals(null, a);
			assertEquals(3214.0, a.getData()[1], 0.0);
			assertEquals(0.0, a.getData()[0], 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void isEmptyNanTransposed() {
		try {

			MatrixBlock mb = new MatrixBlock(10, 5, Double.NaN);

			ReaderColumnSelection reader = ReaderColumnSelection.createReader(mb, ColIndexFactory.create(2), true, 0,
				mb.getNumRows());
			assertEquals(null, reader.nextRow());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void isNaNTransposed() {
		try {

			MatrixBlock mb = new MatrixBlock(10, 5, Double.NaN);
			mb.setValue(1, 1, 3214);

			ReaderColumnSelection reader = ReaderColumnSelection.createReader(mb, ColIndexFactory.create(2), true, 0,
				mb.getNumRows());
			DblArray a = reader.nextRow();
			assertNotEquals(null, a);
			assertEquals(3214.0, a.getData()[1], 0.0);
			assertEquals(0.0, a.getData()[0], 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void isEmptyNanMultiBlock() {
		try {

			MatrixBlock mb = ReadersTestCompareReaders.createMock(new MatrixBlock(10, 5, Double.NaN));

			ReaderColumnSelection reader = ReaderColumnSelection.createReader(mb, ColIndexFactory.create(2), false, 0,
				mb.getNumRows());
			assertEquals(null, reader.nextRow());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void isNaNMultiBlock() {
		try {

			MatrixBlock mb = ReadersTestCompareReaders.createMock(new MatrixBlock(10, 5, Double.NaN));
			mb.setValue(1, 1, 3214);

			ReaderColumnSelection reader = ReaderColumnSelection.createReader(mb, ColIndexFactory.create(2), false, 0,
				mb.getNumRows());
			DblArray a = reader.nextRow();
			assertNotEquals(null, a);
			assertEquals(3214.0, a.getData()[1], 0.0);
			assertEquals(0.0, a.getData()[0], 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void isEmptyNanMultiBlockTransposed() {
		try {

			MatrixBlock mb = ReadersTestCompareReaders.createMock(new MatrixBlock(10, 5, Double.NaN));

			ReaderColumnSelection reader = ReaderColumnSelection.createReader(mb, ColIndexFactory.create(2), true, 0,
				mb.getNumRows());
			assertEquals(null, reader.nextRow());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void isNaNMultiBlockTransposed() {
		try {

			MatrixBlock mb = ReadersTestCompareReaders.createMock(new MatrixBlock(10, 5, Double.NaN));
			mb.setValue(1, 1, 3214);

			ReaderColumnSelection reader = ReaderColumnSelection.createReader(mb, ColIndexFactory.create(2), true, 0,
				mb.getNumRows());
			DblArray a = reader.nextRow();
			assertNotEquals(null, a);
			assertEquals(3214.0, a.getData()[1], 0.0);
			assertEquals(0.0, a.getData()[0], 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void isNanSparseBlock() {
		MatrixBlock mbs = new MatrixBlock(10, 10, true);
		mbs.setValue(1, 1, 3214);
		mbs.setValue(0, 0, Double.NaN);
		mbs.setValue(0, 1, Double.NaN);
		mbs.setValue(1, 0, Double.NaN);

		ReaderColumnSelection reader = ReaderColumnSelection.createReader(mbs, ColIndexFactory.create(2), false, 0,
			mbs.getNumRows());

		DblArray a = reader.nextRow();
		assertNotEquals(null, a);
		assertEquals(3214.0, a.getData()[1], 0.0);
		assertEquals(0.0, a.getData()[0], 0.0);
		assertEquals(null, reader.nextRow());
	}

	@Test
	// for now ignore.. i need a better way of reading matrices containing Nan Becuase the check is very expensive
	@Ignore 
	public void isNanSparseBlockTransposed() {
		MatrixBlock mbs = new MatrixBlock(10, 10, true);
		mbs.setValue(1, 1, 3214);
		mbs.setValue(0, 0, Double.NaN);
		mbs.setValue(0, 1, Double.NaN);
		mbs.setValue(1, 0, Double.NaN);

		ReaderColumnSelection reader = ReaderColumnSelection.createReader(mbs, ColIndexFactory.create(2), true, 0,
			mbs.getNumRows());

		DblArray a = reader.nextRow();
		assertNotEquals(null, a);
		assertEquals(3214.0, a.getData()[1], 0.0);
		assertEquals(0.0, a.getData()[0], 0.0);
		assertEquals(null, reader.nextRow());
	}
}
