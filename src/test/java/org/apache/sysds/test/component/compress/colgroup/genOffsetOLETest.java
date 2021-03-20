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

package org.apache.sysds.test.component.compress.colgroup;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.ColGroupOLE;
import org.junit.Test;

public class genOffsetOLETest {

	@Test
	public void testEmpty() {
		int[] offsets = new int[0];
		int len = 0;
		ColGroupOLE.genOffsetBitmap(offsets, len);
	}

	@Test
	public void testSingleElement_01() {
		int[] offsets = new int[1];
		int len = 1;
		offsets[0] = 5;
		char[] res = ColGroupOLE.genOffsetBitmap(offsets, len);
		assertArrayEquals(new char[] {1, 5}, res);
	}

	@Test
	public void testSingleElement_02() {
		int[] offsets = new int[1];
		int len = 1;
		offsets[0] = 65535;
		int[] res = conv(ColGroupOLE.genOffsetBitmap(offsets, len));
		assertArrayEquals(new int[] {0, 1, 0}, res);
	}

	@Test
	public void testSingleElement_03() {
		int[] offsets = new int[1];
		int len = 1;
		offsets[0] = 65536;
		int[] res = conv(ColGroupOLE.genOffsetBitmap(offsets, len));
		assertArrayEquals(new int[] {0, 1, 1}, res);
	}

	@Test
	public void testSingleElement_04() {
		int[] offsets = new int[1];
		int len = 1;
		offsets[0] = 65534;
		char[] res = ColGroupOLE.genOffsetBitmap(offsets, len);
		assertArrayEquals(new char[] {1, 65534}, res);
	}

	@Test
	public void testTwoElements_01() {
		int[] offsets = new int[2];
		int len = 2;
		offsets[0] = 0;
		offsets[1] = 65536;
		int[] res = conv(ColGroupOLE.genOffsetBitmap(offsets, len));
		assertArrayEquals(new int[] {1, 0, 1, 1}, res);
	}

	@Test
	public void testTwoElements_02() {
		int[] offsets = new int[2];
		int len = 2;
		offsets[0] = 65536;
		offsets[1] = 65536 + 1;
		int[] res = conv(ColGroupOLE.genOffsetBitmap(offsets, len));
		assertArrayEquals(new int[] {0, 2, 1, 2}, res);
	}

	@Test
	public void encodeChar() {
		char v = (char) (CompressionSettings.BITMAP_BLOCK_SZ % (CompressionSettings.BITMAP_BLOCK_SZ));
		assertEquals(0, (int) v);
	}

	private static int[] conv(char[] i) {
		int[] o = new int[i.length];
		int k = 0;
		for(char ii : i)
			o[k++] = ii;
		return o;
	}
}
