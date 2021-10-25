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

package org.apache.sysds.test.component.compress.bitmap;

import static org.junit.Assert.assertEquals;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.bitmap.ABitmap;
import org.apache.sysds.runtime.compress.bitmap.BitmapEncoder;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class BitMapSparseTest {
	protected static final Log LOG = LogFactory.getLog(BitMapSparseTest.class.getName());

	private final MatrixBlock mb;

	public BitMapSparseTest() {
		mb = new MatrixBlock(10, 100, true);
		mb.allocateSparseRowsBlock();
	}

	@Test
	public void constructBitMap() {
		mb.reset();
		final int[] colIndexes = new int[] {0};
		for(int i = 0; i < 10; i++){
			double v = i % 3;
			if(v != 0)
				mb.setValue(i, 0, i % 3);
		}
		ABitmap m = BitmapEncoder.extractBitmap(colIndexes, mb, false, 3, false);
		assertEquals(m.containsZero(), true);
		assertEquals(m.getNumColumns(), 1);
		assertEquals(m.getNumValues(), 2);
		assertEquals(m.getNumOffsets(), 6);
		assertEquals(m.getNumZeros(), 4);
	}

	@Test
	public void constructBitMap_02() {
		mb.reset();
		final int[] colIndexes = new int[] {0};
		for(int i = 0; i < 10; i++){
			double v = i % 3;
			if(v != 0)
				mb.setValue(i, 0, i % 3);
		}
		ABitmap m = BitmapEncoder.extractBitmap(colIndexes, mb, false, 3, false);
		assertEquals(m.containsZero(), true);
		assertEquals(m.getNumColumns(), 1);
		assertEquals(m.getNumValues(), 2);
		assertEquals(m.getNumOffsets(), 6);
		assertEquals(m.getNumZeros(), 4);
	}

	@Test
	public void constructBitMap_transposed_emptyMatrix() {
		mb.reset();
		mb.examSparsity();
		final int[] colIndexes = new int[] {0};

		ABitmap m = BitmapEncoder.extractBitmap(colIndexes, mb, true, 3, false);
		assertEquals(m, null);
	}

	@Test
	public void constructBitMap_transposed_emptyRow() {
		mb.reset();
		mb.examSparsity();
		final int[] colIndexes = new int[] {0};
		mb.appendValue(1, 1, 3241);
		ABitmap m = BitmapEncoder.extractBitmap(colIndexes, mb, true, 3, false);
		assertEquals(m, null);
	}
}
