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
package org.apache.sysds.test.component.frame;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.lib.FrameFromMatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class FrameFromMatrixBlockTest {

	protected static final Log LOG = LogFactory.getLog(FrameFromMatrixBlockTest.class.getName());

	@Test
	public void toBoolean() {
		try {
			MatrixBlock mb = new MatrixBlock(10, 3, 1.0);
			FrameBlock fb = FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.BOOLEAN, 1);
			verifyEquivalence(mb, fb, ValueType.BOOLEAN);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed");
		}
	}

	@Test
	public void toBooleanEmpty() {
		MatrixBlock mb = new MatrixBlock(10, 3, 0.0);
		FrameBlock fb = FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.BOOLEAN, 1);
		verifyEquivalence(mb, fb, ValueType.BOOLEAN);
	}

	@Test
	public void toBooleanSparse() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(100, 100, 1, 1, 0.2, 213);
		FrameBlock fb = FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.BOOLEAN, 1);
		verifyEquivalence(mb, fb, ValueType.BOOLEAN);
	}

	@Test
	public void toBooleanVerySparse() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(100, 100, 1, 1, 0.001, 213);
		FrameBlock fb = FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.BOOLEAN, 1);
		verifyEquivalence(mb, fb, ValueType.BOOLEAN);
	}

	@Test
	public void singleColShortcut() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(100, 1, 0, 1, 0.2, 213);
		FrameBlock fb = FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.FP64, 1);
		verifyEquivalence(mb, fb, ValueType.FP64);
	}

	@Test
	public void singleColShortcutToBoolean() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(100, 1, 1, 1, 0.2, 213);
		FrameBlock fb = FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.BOOLEAN, 1);
		verifyEquivalence(mb, fb, ValueType.BOOLEAN);
	}

	@Test
	public void toFloatDense() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(100, 10, 0, 1, 1.0, 213);
		FrameBlock fb = FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.FP64, 1);
		verifyEquivalence(mb, fb, ValueType.FP64);
	}

	@Test
	public void toFloatDenseParallel() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(1000, 100, 0, 1, 1.0, 213);
		FrameBlock fb = FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.FP64, 4);
		verifyEquivalence(mb, fb, ValueType.FP64);
	}

	@Test
	public void toBooleanDenseParallel() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(1000, 100, 1, 1, 0.5, 213);
		FrameBlock fb = FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.BOOLEAN, 4);
		verifyEquivalence(mb, fb, ValueType.BOOLEAN);
	}

	@Test
	public void toFloatDenseMultiBlock() {
		MatrixBlock mb = mock(TestUtils.generateTestMatrixBlock(100, 10, 0, 1, 1.0, 213));
		FrameBlock fb = FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.FP64, 1);
		verifyEquivalence(mb, fb, ValueType.FP64);
	}

	@Test
	public void toFloatDenseMultiBlockParallel() {
		MatrixBlock mb = mock(TestUtils.generateTestMatrixBlock(100, 10, 0, 1, 1.0, 213));
		FrameBlock fb = FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.FP64, 4);
		verifyEquivalence(mb, fb, ValueType.FP64);
	}

	@Test
	public void toBooleanDenseMultiBlock() {
		MatrixBlock mb = mock(TestUtils.generateTestMatrixBlock(100, 10, 1, 1, 0.7, 213));
		FrameBlock fb = FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.BOOLEAN, 1);
		verifyEquivalence(mb, fb, ValueType.BOOLEAN);
	}

	@Test
	public void shortcutEmpty() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(100, 10, 0, 0, 1.0, 213);
		assertTrue(mb.isEmpty());
		FrameBlock fb = FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.FP64, 1);
		verifyEquivalence(mb, fb, ValueType.FP64);
	}

	@Test
	public void timeChange() {
		// MatrixBlock mb = TestUtils.generateTestMatrixBlock(64000, 2000, 1, 1, 0.5, 2340);

		// for(int i = 0; i < 10; i++) {
		// 	Timing time = new Timing(true);
		// 	FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.BOOLEAN, 1);
		// 	LOG.error(time.stop());
		// }

		// for(int i = 0; i < 10; i++) {
		// 	Timing time = new Timing(true);
		// 	FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.BOOLEAN, 16);
		// 	LOG.error(time.stop());
		// }

		// for(int i = 0; i < 10; i ++){
		// Timing time = new Timing(true);
		// FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.FP64, 1);
		// LOG.error(time.stop());
		// }

		// for(int i = 0; i < 10; i++) {
		// Timing time = new Timing(true);
		// FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.FP64, 16);
		// LOG.error(time.stop());
		// }
	}

	private void verifyEquivalence(MatrixBlock mb, FrameBlock fb, ValueType vt) {
		int nRow = mb.getNumRows();
		int nCol = mb.getNumColumns();
		assertEquals(mb.getNumColumns(), fb.getSchema().length);
		for(int i = 0; i < nCol; i++)
			assertTrue(fb.getColumn(i).getValueType() == vt);

		for(int i = 0; i < nRow; i++)
			for(int j = 0; j < nCol; j++)
				assertEquals(mb.getValue(i, j), fb.getDouble(i, j), 0.0000001);

	}

	private MatrixBlock mock(MatrixBlock m) {
		MatrixBlock ret = new MatrixBlock(m.getNumRows(), m.getNumColumns(),
			new DenseBlockFP64Mock(new int[] {m.getNumRows(), m.getNumColumns()}, m.getDenseBlockValues()));
		ret.setNonZeros(m.getNumRows() * m.getNumColumns());
		return ret;

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
			return 1;
		}
	}

}
