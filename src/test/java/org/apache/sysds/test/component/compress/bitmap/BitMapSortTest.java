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

import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.bitmap.ABitmap;
import org.apache.sysds.runtime.compress.bitmap.BitmapEncoder;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class BitMapSortTest {
	protected static final Log LOG = LogFactory.getLog(BitMapSortTest.class.getName());

	@Parameters
	public static Collection<Object[]> data() {
		List<Object[]> tests = new ArrayList<>();
		tests.add(new Object[] {1, 100, 3, 142});
		tests.add(new Object[] {1, 500, 6, 142});
		tests.add(new Object[] {1, 1000, 10, 142});
		tests.add(new Object[] {2, 1000, 3, 4});
		return tests;
	}

	private final MatrixBlock mb;
	private final int[] colIndexes;

	public BitMapSortTest(int cols, int rows, int nUniqueMax, int seed) {
		colIndexes = Util.genColsIndices(cols);
		mb = new MatrixBlock(rows, cols, true);
		mb.allocateDenseBlock();
		double[] vals = mb.getDenseBlockValues();
		Random r = new Random(seed);
		double mul = nUniqueMax * (nUniqueMax + 1);
		for(int i = 0; i < cols * rows; i++)
			vals[i] = 1 + Math.floor(Math.pow(r.nextDouble() * mul, 0.5));

		mb.setNonZeros(cols * rows);
	}

	@Test
	public void sortBitmap() {
		ABitmap m = BitmapEncoder.extractBitmap(colIndexes, mb, false, 2, true);
		verifySortedOffsets(m);
	}

	@Test 
	public void toStringBitmap(){
		// just verify that it does not crash because of toString
		BitmapEncoder.extractBitmap(colIndexes, mb, false, 2, true).toString();
	}

	private String getLengthsString(ABitmap m) {
		StringBuilder sb = new StringBuilder();
		for(IntArrayList of : m.getOffsetList())
			sb.append(of.size() + " - ");
		return sb.toString();
	}

	private void verifySortedOffsets(ABitmap m) {
		IntArrayList[] offsets = m.getOffsetList();
		for(int i = 0; i < offsets.length - 1; i++)
			if(offsets[i].size() < offsets[i + 1].size())
				fail("The offsets are not sorted \n" + getLengthsString(m));
	}
}
