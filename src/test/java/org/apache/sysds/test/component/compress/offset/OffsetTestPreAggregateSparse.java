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
import java.util.Arrays;
import java.util.Collection;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.util.Precision;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetByte;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetChar;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory.OFF_TYPE;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public abstract class OffsetTestPreAggregateSparse {
	protected static final Log LOG = LogFactory.getLog(OffsetTestPreAggregateSparse.class.getName());

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
			tests.add(new Object[] {new int[] {4, 6, 11, 14, 24, 34, 38, 40, 43, 46, 47, 52, 53, 64, 69, 70, 71, 72, 76,
				77, 80, 83, 94, 109, 112, 114, 125, 133, 138, 142, 144, 147, 158, 159, 162, 167, 171, 177, 186, 198, 203,
				204, 207, 209, 212, 219, 221, 224, 225, 227, 229, 231, 234, 242, 252, 253, 255, 257, 263, 271, 276, 277,
				288, 296, 297, 300, 306, 313, 320, 321, 336, 358, 365, 385, 387, 391, 397, 399, 408, 414, 416, 417, 419,
				425, 429, 438, 441, 445, 459, 477, 482, 483, 499}, t});
		}
		return tests;
	}

	public OffsetTestPreAggregateSparse(int[] data, OFF_TYPE type) {
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
		this.leftM = TestUtils.generateTestMatrixBlock(2, data[data.length - 1] + 100, 100, 200, 0.35, 23152);
		this.s = sumIndexes();
	}

	@Test
	public void preAggByteMapFirstRow() {
		preAggMapRow(0);
	}

	@Test
	public void preAggByteMapSecondRow() {
		preAggMapRow(1);
	}

	protected abstract void preAggMapRow(int row);

	protected void verifyPreAggMapRow(double[] preAV, int row) {
		if(preAV[0] != s[row]) {
			fail("\nThe preaggregate result is not the sum! : " + a.getClass().getSimpleName() + "  " + preAV[0] + " vs "
				+ s[row] + "\n full agg: " + Arrays.toString(preAV));
		}
	}

	@Test(expected = NotImplementedException.class)
	public abstract void preAggMapAllRows();

	protected void verifyPreAggMapAllRow(double[] preAV) {
		if(preAV[1] != 0)
			fail("\naggregate to wrong index");
		if(preAV[3] != 0)
			fail("\naggregate to wrong index");
		if(!Precision.equals(preAV[0], s[0], eps))
			fail("\nThe preaggregate result is not the sum!: " + preAV[0] + "vs" + s[0]);
		if(!Precision.equals(preAV[2], s[1], eps))
			fail("\nThe preaggregate result is not the sum!: " + preAV[2] + "vs" + s[1]);
	}

	private final double[] sumIndexes() {
		double[] ret = new double[leftM.getNumRows()];
		SparseBlock sb = leftM.getSparseBlock();
		for(int j = 0; j < leftM.getNumRows(); j++) {
			if(sb.isEmpty(j))
				continue;
			final int apos = sb.pos(j);
			final int alen = sb.size(j) + apos;
			final int[] aix = sb.indexes(j);
			final double[] avals = sb.values(j);
			int dx = 0;
			for(int i = apos; i < alen && dx < data.length; i++) {
				while(dx < data.length && data[dx] < aix[i])
					dx++;
				if(dx < data.length && data[dx] == aix[i])
					ret[j] += avals[i];
			}
		}
		return ret;
	}

}
