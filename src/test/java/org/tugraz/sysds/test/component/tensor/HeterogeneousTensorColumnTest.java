/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.tugraz.sysds.test.component.tensor;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.data.HeterogTensor;
import org.tugraz.sysds.runtime.data.HomogTensor;

import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.LongStream;


public class HeterogeneousTensorColumnTest {
	private static int DIM0 = 3, DIM1 = 5, DIM2 = 7;

	@Test
	public void testIndexHeterogTensor2FP32FillCol() {
		HeterogTensor tb = getHeterogTensor2(ValueType.FP32);
		checkCol(fillCol(tb));
	}

	@Test
	public void testIndexHeterogTensor2FP64FillCol() {
		HeterogTensor tb = getHeterogTensor2(ValueType.FP64);
		checkCol(fillCol(tb));
	}

	@Test
	public void testIndexHeterogTensor2BoolFillCol() {
		HeterogTensor tb = getHeterogTensor2(ValueType.BOOLEAN);
		checkCol(fillCol(tb));
	}

	@Test
	public void testIndexHeterogTensor2StringFillCol() {
		HeterogTensor tb = getHeterogTensor2(ValueType.STRING);
		checkCol(fillCol(tb));
	}

	@Test
	public void testIndexHeterogTensor2Int32FillCol() {
		HeterogTensor tb = getHeterogTensor2(ValueType.INT32);
		checkCol(fillCol(tb));
	}

	@Test
	public void testIndexHeterogTensor2Int64FillCol() {
		HeterogTensor tb = getHeterogTensor2(ValueType.INT64);
		checkCol(fillCol(tb));
	}

	@Test
	public void testIndexHeterogTensor3FP32FillCol() {
		HeterogTensor tb = getHeterogTensor3(ValueType.FP32);
		checkCol(fillCol(tb));
	}

	@Test
	public void testIndexHeterogTensor3FP64FillCol() {
		HeterogTensor tb = getHeterogTensor3(ValueType.FP64);
		checkCol(fillCol(tb));
	}

	@Test
	public void testIndexHeterogTensor3BoolFillCol() {
		HeterogTensor tb = getHeterogTensor3(ValueType.BOOLEAN);
		checkCol(fillCol(tb));
	}

	@Test
	public void testIndexHeterogTensor3StringFillCol() {
		HeterogTensor tb = getHeterogTensor3(ValueType.STRING);
		checkCol(fillCol(tb));
	}

	@Test
	public void testIndexHeterogTensor3Int32FillCol() {
		HeterogTensor tb = getHeterogTensor3(ValueType.INT32);
		checkCol(fillCol(tb));
	}

	@Test
	public void testIndexHeterogTensor3Int64FillCol() {
		HeterogTensor tb = getHeterogTensor3(ValueType.INT64);
		checkCol(fillCol(tb));
	}

	@Test
	public void testIndexHeterogTensorFP32AppendCols() {
		HeterogTensor tb = getHeterogTensorByAppendCols(ValueType.FP32);
		checkCol(tb);
	}

	@Test
	public void testIndexHeterogTensorFP64AppendCols() {
		HeterogTensor tb = getHeterogTensorByAppendCols(ValueType.FP64);
		checkCol(tb);
	}

	@Test
	public void testIndexHeterogTensorBoolAppendCols() {
		HeterogTensor tb = getHeterogTensorByAppendCols(ValueType.BOOLEAN);
		checkCol(tb);
	}

	@Test
	public void testIndexHeterogTensorStringAppendCols() {
		HeterogTensor tb = getHeterogTensorByAppendCols(ValueType.STRING);
		checkCol(tb);
	}

	@Test
	public void testIndexHeterogTensorInt32AppendCols() {
		HeterogTensor tb = getHeterogTensorByAppendCols(ValueType.INT32);
		checkCol(tb);
	}

	@Test
	public void testIndexHeterogTensorInt64AppendCols() {
		HeterogTensor tb = getHeterogTensorByAppendCols(ValueType.INT64);
		checkCol(tb);
	}

	private HeterogTensor getHeterogTensor2(ValueType vt) {
		return new HeterogTensor(vt, new int[]{DIM0, DIM1});
	}

	private HeterogTensor getHeterogTensor3(ValueType vt) {
		return new HeterogTensor(vt, new int[]{DIM0, DIM1, DIM2});
	}

	private HeterogTensor fillCol(HeterogTensor tb) {
		int length;
		if (tb.getNumDims() == 3) {
			length = DIM0 * DIM2;
		}
		else { //num dims = 2
			length = DIM0;
		}
		for (int c = 0; c < DIM1; c++) {
			switch (tb.getColValueType(c)) {
				case INT64:
				case INT32:
					tb.fillCol(c, LongStream.range(0, length).toArray());
					break;
				case STRING:
					String[] col = new String[length];
					Arrays.fill(col, "test");
					tb.fillCol(c, col);
					break;
				default:
					tb.fillCol(c, IntStream.range(0, length).mapToDouble((i) -> (double)i).toArray());
					break;
			}
		}
		return tb;
	}

	private HeterogTensor getHeterogTensorByAppendCols(ValueType vt) {
		HeterogTensor tb = new HeterogTensor(vt, new int[]{DIM0, 0, DIM2});
		for (int c = 0; c < DIM1; c++) {
			HomogTensor ht = new HomogTensor(vt, new int[]{DIM0, 1, DIM2}, false);
			int[] ix = new int[ht.getNumDims()];
			for (int i = 0; i < ht.getLength(); i++) {
				switch (vt) {
					case INT64:
					case INT32:
						ht.set(ix, i);
						break;
					case STRING:
						ht.set(ix, "test");
						break;
					default:
						ht.set(ix, (double) i);
						break;
				}
				ht.getNextIndexes(ix);
			}
			tb = tb.appendCol(ht);
		}
		return tb;
	}

	private void checkCol(HeterogTensor tb) {
		int length;
		if (tb.getNumDims() == 3) {
			length = DIM0 * DIM2;
		}
		else { //num dims = 2
			length = DIM0;
		}
		for (int c = 0; c < DIM1; c++) {
			HomogTensor colTensor = tb.getCol(c);
			int[] ix = new int[colTensor.getNumDims()];
			for (int i = 0; i < length; i++) {
				switch (tb.getColValueType(c)) {
					case INT64:
					case INT32:
						Assert.assertEquals(i, colTensor.getLong(ix));
						break;
					case STRING:
						Assert.assertEquals("test", colTensor.getString(ix));
						break;
					case BOOLEAN:
						Assert.assertEquals(i > 0 ? 1 : 0, colTensor.getLong(ix));
						break;
					default:
						Assert.assertEquals(i, colTensor.get(ix), 0);
						break;
				}
				colTensor.getNextIndexes(ix);
			}
		}
	}
}
