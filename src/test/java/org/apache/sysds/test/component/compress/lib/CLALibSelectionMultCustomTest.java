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

package org.apache.sysds.test.component.compress.lib;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.apache.sysds.runtime.compress.lib.CLALibSelectionMult;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class CLALibSelectionMultCustomTest {

	@Test
	public void isSelectionEmpty() {
		MatrixBlock mb = new MatrixBlock(10, 10, false);
		assertFalse(CLALibSelectionMult.isSelectionMatrix(mb));
	}

	@Test
	public void isSelectionOneCell() {
		MatrixBlock mb = new MatrixBlock(10, 10, true);
		mb.allocateSparseRowsBlock();
		mb.appendValue(1, 1, 1);
		assertTrue(CLALibSelectionMult.isSelectionMatrix(mb));
	}

	@Test
	public void isSelectionOneCellIncorrectValue() {
		MatrixBlock mb = new MatrixBlock(10, 10, true);
		mb.allocateSparseRowsBlock();
		mb.appendValue(1, 1, 2);
		assertFalse(CLALibSelectionMult.isSelectionMatrix(mb));
	}

	@Test
	public void isSelectionOneCellCSR() {
		MatrixBlock mb = new MatrixBlock(10, 10, true);
		mb.allocateSparseRowsBlock();
		mb.appendValue(1, 1, 1);
		SparseBlockCSR sb = new SparseBlockCSR(mb.getSparseBlock());
		mb.setSparseBlock(sb);
		assertTrue(CLALibSelectionMult.isSelectionMatrix(mb));
	}

	@Test
	public void isSelectionOneCellCSRIncorrectValue() {
		MatrixBlock mb = new MatrixBlock(10, 10, true);
		mb.allocateSparseRowsBlock();
		mb.appendValue(1, 1, 2);
		SparseBlockCSR sb = new SparseBlockCSR(mb.getSparseBlock());
		mb.setSparseBlock(sb);
		assertFalse(CLALibSelectionMult.isSelectionMatrix(mb));
	}

	@Test
	public void isSelectionTwoCellsOneRow() {
		MatrixBlock mb = new MatrixBlock(10, 10, true);
		mb.allocateSparseRowsBlock();
		mb.appendValue(1, 1, 1);
		mb.appendValue(1, 2, 1);
		assertFalse(CLALibSelectionMult.isSelectionMatrix(mb));
	}

	@Test
	public void isSelectionTwoCellsTwoRows() {
		MatrixBlock mb = new MatrixBlock(10, 10, true);
		mb.allocateSparseRowsBlock();
		mb.appendValue(1, 1, 1);
		mb.appendValue(0, 1, 1);
		assertTrue(CLALibSelectionMult.isSelectionMatrix(mb));
	}

	@Test
	public void isSelectionTwoCellsTwoRowsInvalidValue() {
		MatrixBlock mb = new MatrixBlock(10, 10, true);
		mb.allocateSparseRowsBlock();
		mb.appendValue(1, 1, 1);
		mb.appendValue(0, 1, 2);
		assertFalse(CLALibSelectionMult.isSelectionMatrix(mb));
	}

	@Test
	public void isSelectionMorePointsThanRows() {
		MatrixBlock mb = new MatrixBlock(2, 2, true);
		mb.allocateSparseRowsBlock();
		mb.appendValue(1, 1, 1);
		mb.appendValue(0, 1, 1);
		mb.appendValue(0, 0, 1);
		assertFalse(CLALibSelectionMult.isSelectionMatrix(mb));
	}

	@Test
	public void isSelectionDenseBlock() {
		MatrixBlock mb = new MatrixBlock(2, 2, false);
		mb.allocateDenseBlock();
		// mb.allocateSparseRowsBlock();
		mb.appendValue(1, 1, 1);
		mb.appendValue(0, 1, 1);
		// mb.appendValue(2, 2, 1);
		assertFalse(CLALibSelectionMult.isSelectionMatrix(mb));
	}

	@Test
	public void selectionError() {
		Exception e = assertThrows(Exception.class, () -> CLALibSelectionMult.leftSelection(null, null, null, 1));
		assertTrue(e.getMessage().contains("Failed left selection Multiplication"));
	}
}
