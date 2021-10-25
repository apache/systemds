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
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.bitmap.ABitmap;
import org.apache.sysds.runtime.compress.bitmap.BitmapEncoder;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class BitMapTest {
	protected static final Log LOG = LogFactory.getLog(BitMapTest.class.getName());

	private final MatrixBlock mb;
	private final int[] colIndexes;

	public BitMapTest() {
		colIndexes = new int[] {0};
		mb = new MatrixBlock(10, 1, true);
		mb.allocateDenseBlock();
	}

	@Test
	public void constructBitMap() {
		for(int i = 0; i < 10; i++)
			mb.setValue(i, 0, 1 + i % 3);
		ABitmap m = BitmapEncoder.extractBitmap(colIndexes, mb, false, 3, false);
		assertEquals(m.containsZero(), false);
		assertEquals(m.getNumColumns(), 1);
		assertEquals(m.getNumValues(), 3);
		assertEquals(m.getNumOffsets(), 10);
		assertEquals(m.getNumZeros(), 0);
	}

	@Test
	public void constructBitMapWithZeros() {
		for(int i = 0; i < 10; i++)
			mb.setValue(i, 0, i % 3);
		ABitmap m = BitmapEncoder.extractBitmap(colIndexes, mb, false, 2, false);
		assertEquals(m.containsZero(), true);
		assertEquals(m.getNumColumns(), 1);
		assertEquals(m.getNumValues(), 2);
		assertEquals(m.getNumOffsets(), 6);
		assertEquals(m.getNumZeros(), 4);
	}

	@Test
	public void sortBitmap() {
		for(int i = 0; i < 10; i++)
			mb.setValue(i, 0, i > 7 ? 1 : 2);
		ABitmap m = BitmapEncoder.extractBitmap(colIndexes, mb, false, 2, true);
		assertEquals(m.containsZero(), false);
		assertEquals(m.getNumColumns(), 1);
		assertEquals(m.getNumValues(), 2);
		assertEquals(m.getNumOffsets(), 10);
		assertEquals(m.getNumZeros(), 0);
		verifySortedOffsets(m);
	}

	@Test
	public void returnNullOnEmptyInput() {
		ABitmap m = BitmapEncoder.extractBitmap(colIndexes, new MatrixBlock(10, 10, true), false, 2, true);
		assertTrue(m == null);
	}

	@Test
	public void returnNullOnNullInput() {
		ABitmap m = BitmapEncoder.extractBitmap(colIndexes, null, false, 2, true);
		assertTrue(m == null);
	}

	private void verifySortedOffsets(ABitmap m) {
		IntArrayList[] offsets = m.getOffsetList();
		for(int i = 0; i < offsets.length - 1; i++)
			if(offsets[i].size() < offsets[i + 1].size())
				fail("The offsets are not sorted \n" + m);

	}
}
