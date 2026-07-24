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
import static org.junit.Assert.assertNotNull;

import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelection;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class ReaderColumnSelectionSparseDeltaTest {

	@Test
	public void testSparseDeltaReaderEmptyRowSkips() {
		MatrixBlock mb = new MatrixBlock(4, 3, true);
		mb.allocateSparseRowsBlock();
		
		mb.appendValue(0, 0, 1.0);
		mb.appendValue(2, 0, 5.0);
		mb.appendValue(3, 2, 10.0);

		IColIndex colIndexes = ColIndexFactory.create(new int[] {0});
		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(mb, colIndexes, false);
		
		DblArray row0 = reader.nextRow();
		assertEquals(1.0, row0.getData()[0], 0.0);
		
		DblArray row1 = reader.nextRow();
		assertEquals(-1.0, row1.getData()[0], 0.0);
		
		DblArray row2 = reader.nextRow();
		assertEquals(5.0, row2.getData()[0], 0.0);
		
		DblArray row3 = reader.nextRow();
		assertEquals(-5.0, row3.getData()[0], 0.0);
	}

	@Test
	public void testSparseDeltaReaderTargetSmallerThanSparse() {
		MatrixBlock mb = new MatrixBlock(2, 5, true);
		mb.allocateSparseRowsBlock();
		
		mb.appendValue(0, 1, 10.0);
		mb.appendValue(0, 3, 20.0);
		
		mb.appendValue(1, 2, 30.0);
		mb.appendValue(1, 4, 40.0);

		IColIndex colIndexes = ColIndexFactory.create(new int[] {0, 2});
		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(mb, colIndexes, false);
		
		DblArray row0 = reader.nextRow();
		assertNotNull(row0);
		assertEquals(0.0, row0.getData()[0], 0.0);
		assertEquals(0.0, row0.getData()[1], 0.0);
		
		DblArray row1 = reader.nextRow();
		assertNotNull(row1);
		assertEquals(0.0, row1.getData()[0], 0.0);
		assertEquals(30.0, row1.getData()[1], 0.0);
	}

	@Test
	public void testSparseDeltaReaderColumnIndexAheadOfSparse() {
		MatrixBlock mb = new MatrixBlock(2, 10, true);
		mb.allocateSparseRowsBlock();
		
		mb.appendValue(0, 1, 10.0);
		mb.appendValue(0, 2, 15.0);
		
		mb.appendValue(1, 1, 20.0);
		mb.appendValue(1, 2, 25.0);
		mb.appendValue(1, 3, 30.0);

		IColIndex colIndexes = ColIndexFactory.create(new int[] {3, 4});
		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(mb, colIndexes, false);
		
		DblArray row0 = reader.nextRow();
		assertNotNull(row0);
		assertEquals(0.0, row0.getData()[0], 0.0);
		assertEquals(0.0, row0.getData()[1], 0.0);
		
		DblArray row1 = reader.nextRow();
		assertNotNull(row1);
		assertEquals(30.0, row1.getData()[0], 0.0);
		assertEquals(0.0, row1.getData()[1], 0.0);
	}

	@Test
	public void testSparseDeltaReaderColumnIndexBehindSparse() {
		MatrixBlock mb = new MatrixBlock(2, 10, true);
		mb.allocateSparseRowsBlock();
		
		mb.appendValue(0, 3, 10.0);
		mb.appendValue(0, 5, 20.0);
		
		mb.appendValue(1, 1, 30.0);
		mb.appendValue(1, 7, 40.0);

		IColIndex colIndexes = ColIndexFactory.create(new int[] {1, 3, 5});
		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(mb, colIndexes, false);
		
		DblArray row0 = reader.nextRow();
		assertNotNull(row0);
		assertEquals(0.0, row0.getData()[0], 0.0);
		assertEquals(10.0, row0.getData()[1], 0.0);
		assertEquals(20.0, row0.getData()[2], 0.0);
		
		DblArray row1 = reader.nextRow();
		assertNotNull(row1);
		assertEquals(30.0, row1.getData()[0], 0.0);
		assertEquals(-10.0, row1.getData()[1], 0.0);
		assertEquals(-20.0, row1.getData()[2], 0.0);
	}
}

