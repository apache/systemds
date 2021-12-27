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

package org.apache.sysds.test.component.compress.offset;

import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.util.Precision;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetByte;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetChar;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory.OFF_TYPE;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public abstract class OffsetTestPreAggregate {
	protected static final Log LOG = LogFactory.getLog(OffsetTestPreAggregate.class.getName());

	protected static final double eps = 0.00001;

	protected final int[] data;
	protected final AOffset a;

	protected final MatrixBlock leftM;

	// sum of indexes row 1.
	protected final double[] s;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		// It is assumed that the input is in sorted order, all values are positive and there are no duplicates.
		// note that each tests allocate an matrix of two rows, and the last value length.
		// therefore don't make it to large.
		for(OFF_TYPE t : OFF_TYPE.values()) {
			tests.add(new Object[] {new int[] {1, 2}, t});
			tests.add(new Object[] {new int[] {2, 142}, t});
			tests.add(new Object[] {new int[] {142, 421}, t});
			tests.add(new Object[] {new int[] {1, 1023}, t});
			tests.add(new Object[] {new int[] {1023, 1024}, t});
			tests.add(new Object[] {new int[] {1023}, t});
			tests.add(new Object[] {new int[] {0, 1, 2, 3, 4, 5}, t});
			tests.add(new Object[] {new int[] {0}, t});
			tests.add(new Object[] {new int[] {0, 256}, t});
			tests.add(new Object[] {new int[] {0, 254}, t});
			tests.add(new Object[] {new int[] {0, 256 * 2}, t});
			tests.add(new Object[] {new int[] {0, 255 * 2}, t});
			tests.add(new Object[] {new int[] {0, 254 * 2}, t});
			tests.add(new Object[] {new int[] {0, 510, 765}, t});
			tests.add(new Object[] {new int[] {0, 254 * 3}, t});
			tests.add(new Object[] {new int[] {0, 255, 255 * 2, 255 * 3}, t});
			tests.add(new Object[] {new int[] {0, 255 * 2, 255 * 3}, t});
			tests.add(new Object[] {new int[] {0, 255 * 2, 255 * 3, 255 * 10}, t});
			tests.add(new Object[] {new int[] {0, 255 * 3}, t});
			tests.add(new Object[] {new int[] {0, 255 * 4}, t});
			tests.add(new Object[] {new int[] {0, 256 * 3}, t});
			tests.add(new Object[] {new int[] {255 * 3, 255 * 5}, t});
			tests.add(new Object[] {new int[] {0, 1, 2, 3, 255 * 4, 1500}, t});
			tests.add(new Object[] {new int[] {0, 1, 2, 3, 4, 5}, t});
			tests.add(new Object[] {new int[] {0, 1, 2, 3, 4, 5, 125, 142, 161, 1661, 2314}, t});
			tests.add(new Object[] {new int[] {51, 4251, Character.MAX_VALUE}, t});
			tests.add(new Object[] {new int[] {0, Character.MAX_VALUE + 10}, t});
			tests.add(new Object[] {new int[] {0, Character.MAX_VALUE + 10, (Character.MAX_VALUE + 10) * 2}, t});
			tests.add(new Object[] {new int[] {4, 6, 11, 14, 24, 34, 38, 40, 43, 46, 47, 52, 53, 64, 69, 70, 71, 72, 76,
				77, 80, 83, 94, 109, 112, 114, 125, 133, 138, 142, 144, 147, 158, 159, 162, 167, 171, 177, 186, 198, 203,
				204, 207, 209, 212, 219, 221, 224, 225, 227, 229, 231, 234, 242, 252, 253, 255, 257, 263, 271, 276, 277,
				288, 296, 297, 300, 306, 313, 320, 321, 336, 358, 365, 385, 387, 391, 397, 399, 408, 414, 416, 417, 419,
				425, 429, 438, 441, 445, 459, 477, 482, 483, 499}, t});
		}
		return tests;
	}

	public OffsetTestPreAggregate(int[] data, OFF_TYPE type) {
		this.data = data;
		switch(type) {
			case BYTE:
				this.a = new OffsetByte(data);
				break;
			case CHAR:
				this.a = new OffsetChar(data);
				break;
			default:
				throw new NotImplementedException("not implemented");
		}

		this.leftM = TestUtils.generateTestMatrixBlock(4, data[data.length - 1] + 100, -1, 100, 1.0, 1342);
		this.s = sumIndexes();
	}

	@Test
	public void preAggByteMapFirstRow() {
		try {
			preAggMapRow(0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void preAggByteMapSecondRow() {
		try {
			preAggMapRow(1);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	protected abstract void preAggMapRow(int row);

	protected void verifyPreAggMapRowByte(double[] preAV, int row) {

		if(preAV[0] != s[row])
			fail("\nThe preaggregate result is not the sum! : " + preAV[0] + " vs " + s[row]);
	}

	@Test
	public void preAggByteMapFirstRowByteAll1() {
		preAggMapRowAll1(0);
	}

	@Test
	public void preAggByteMapSecondRowByteAll1() {
		preAggMapRowAll1(1);
	}

	protected abstract void preAggMapRowAll1(int row);

	protected void verifyPreAggMapRowAllBytes1(double[] preAV, int row) {
		if(preAV[0] != 0)
			fail("\naggregate to wrong index");
		if(preAV[1] != s[row])
			fail("\nThe preaggregate result is not the sum! : " + preAV[0] + " vs " + s[row]);
	}

	@Test
	public void preAggByteMapFirstRowByteOne1() {
		preAggMapRowOne1(0);
	}

	@Test
	public void preAggByteMapSecondRowByteOne1() {
		preAggMapRowOne1(1);
	}

	protected abstract void preAggMapRowOne1(int row);

	protected void verifyPreAggMapRowOne1(double[] preAV, int row) {
		double v = leftM.getValue(row, data[1]);
		if(preAV[1] != v)
			fail("\naggregate to wrong index");
		if(!Precision.equals(preAV[0], s[row] - v, eps))
			fail("\nThe preaggregate result is not the sum! : " + preAV[0] + " vs " + (s[row] - v));
	}

	@Test
	public abstract void preAggMapAllRowsOne1();

	protected void verifyPreAggAllOne1(double[] preAV) {
		double v1 = leftM.getValue(0, data[1]);
		double v2 = leftM.getValue(1, data[1]);
		if(preAV[1] != v1)
			fail("\naggregate to wrong index");
		if(preAV[3] != v2)
			fail("\naggregate to wrong index");
		if(!Precision.equals(preAV[0], s[0] - v1, eps))
			fail("\nThe preaggregate result is not the sum! : " + preAV[0] + " vs " + (s[0] - v1));
		if(!Precision.equals(preAV[2], s[1] - v2, eps))
			fail("\nThe preaggregate result is not the sum! : " + preAV[2] + " vs " + (s[1] - v2));
	}

	@Test
	public void preAggByteMapFirstSubOfRow() {
		preAggMapSubOfRow(0);
	}

	@Test
	public void preAggByteMapSecondSubOfRow() {
		preAggMapSubOfRow(1);
	}

	protected abstract void preAggMapSubOfRow(int row);

	protected void verifyPreAggMapSubOfRow(double[] preAV, int row) {
		double v = leftM.getValue(row, data[1]);
		double v2 = leftM.getValue(row, data[data.length - 1]);
		if(preAV[1] != v)
			fail("\naggregate to wrong index");
		if(!Precision.equals(preAV[0], s[row] - v - v2, eps))
			fail("\nThe preaggregate result is not the sum! : " + preAV[0] + " vs " + (s[row] - v - v2));
	}

	@Test
	public void preAggByteMapFirstSubOfRowV2() {
		preAggMapSubOfRowV2(0, 2);
	}

	@Test
	public void preAggByteMapSecondSubOfRowV2() {
		preAggMapSubOfRowV2(1, 2);
	}

	@Test
	public void preAggByteMapFirstSubOfRowV2V2() {
		preAggMapSubOfRowV2(0, 244);
	}

	@Test
	public void preAggByteMapSecondSubOfRowV2V2() {
		preAggMapSubOfRowV2(1, 244);
	}

	protected abstract void preAggMapSubOfRowV2(int row, int nVal);

	protected void verifyPreAggMapSubOfRowV2(double[] preAV, int row) {
		double v = leftM.getValue(row, data[1]);
		double v2 = leftM.getValue(row, data[data.length - 1]) + leftM.getValue(row, data[data.length - 2]);
		if(preAV[1] != v)
			fail("\naggregate to wrong index");
		if(!Precision.equals(preAV[0], s[row] - v - v2, eps))
			fail("\nThe preaggregate result is not the sum! : " + preAV[0] + " vs " + (s[row] - v - v2));
	}

	@Test
	public void preAggByteMapFirstOutOfRangeBefore() {
		preAggMapOutOfRangeBefore(0);
	}

	@Test
	public void preAggByteMapSecondOutOfRangeBefore() {
		preAggMapOutOfRangeBefore(1);
	}

	protected abstract void preAggMapOutOfRangeBefore(int row);

	@Test
	public void preAggByteMapFirstOutOfRangeAfter() {
		preAggMapOutOfRangeAfter(0);
	}

	@Test
	public void preAggByteMapSecondOutOfRangeAfter() {
		preAggMapOutOfRangeAfter(1);
	}

	protected abstract void preAggMapOutOfRangeAfter(int row);

	@Test
	public void multiRowPreAggRange01() {
		if(data.length > 2) {
			double[] agg = multiRowPreAggRangeSafe(1, 3);
			compareMultiRowAgg(agg, 1, 3);
		}
	}

	@Test
	public void multiRowPreAggRange02() {
		if(data.length > 2) {
			double[] agg = multiRowPreAggRangeSafe(2, 4);
			compareMultiRowAgg(agg, 2, 4);
		}
	}

	@Test
	public void multiRowPreAggRange03() {
		if(data.length > 2) {
			double[] agg = multiRowPreAggRangeSafe(0, 4);
			compareMultiRowAgg(agg, 0, 4);
		}
	}

	@Test
	public void multiRowPreAggRange04() {
		if(data.length > 2) {
			double[] agg = multiRowPreAggRangeSafe(0, 3);
			compareMultiRowAgg(agg, 0, 3);
		}
	}

	protected double[] multiRowPreAggRangeSafe(int rl, int ru) {
		try {
			return multiRowPreAggRange(rl, ru);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.toString());
			return null;
		}
	}

	protected abstract double[] multiRowPreAggRange(int rl, int ru);

	protected void compareMultiRowAgg(double[] agg, int rl, int ru) {
		for(int r = rl, of = 0; r < ru; r++, of++) {
			double v = leftM.getValue(r, data[1]);

			if(agg[of * 2 + 1] != v)
				fail("\naggregate to wrong index");
			if(!Precision.equals(agg[of * 2], s[r] - v, eps))
				fail("\naggregate result is not sum minus value:" + agg[of * 2] + " vs " + (s[r] - v));
		}
	}

	@Test
	public void multiRowPreAggRangeBeforeLast01() {
		try {
			if(data.length > 2) {
				double[] agg = multiRowPreAggRangeBeforeLast(1, 3);
				compareMultiRowAggBeforeLast(agg, 1, 3);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void multiRowPreAggRangeBeforeLast02() {
		try {
			if(data.length > 2) {
				double[] agg = multiRowPreAggRangeBeforeLast(2, 4);
				compareMultiRowAggBeforeLast(agg, 2, 4);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void multiRowPreAggRangeBeforeLast03() {
		try {
			if(data.length > 2) {
				double[] agg = multiRowPreAggRangeBeforeLast(0, 4);
				compareMultiRowAggBeforeLast(agg, 0, 4);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void multiRowPreAggRangeBeforeLast04() {
		try {
			if(data.length > 2) {
				double[] agg = multiRowPreAggRangeBeforeLast(0, 3);
				compareMultiRowAggBeforeLast(agg, 0, 3);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	protected abstract double[] multiRowPreAggRangeBeforeLast(int rl, int ru);

	protected void compareMultiRowAggBeforeLast(double[] agg, int rl, int ru) {
		for(int r = rl, of = 0; r < ru; r++, of++) {
			double v = leftM.getValue(r, data[1]);
			double v2 = leftM.getValue(r, data[data.length - 1]);
			if(agg[of * 2 + 1] != v)
				fail("\naggregate to wrong index");
			if(!Precision.equals(agg[of * 2], s[r] - v - v2, eps))
				fail("\naggregate result is not sum minus value:" + agg[of * 2] + " vs " + (s[r] - v - v2));
		}
	}

	private final double[] sumIndexes() {
		double[] ret = new double[leftM.getNumRows()];
		double[] lmv = leftM.getDenseBlockValues();
		for(int j = 0; j < leftM.getNumRows(); j++) {
			final int off = j * leftM.getNumColumns();
			for(int i = 0; i < data.length; i++)
				ret[j] += lmv[data[i] + off];
		}
		return ret;
	}

}
