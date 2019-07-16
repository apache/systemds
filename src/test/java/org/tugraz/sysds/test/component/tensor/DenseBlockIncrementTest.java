/*
 * Copyright 2018 Graz University of Technology
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
 */

package org.tugraz.sysds.test.component.tensor;

import org.apache.commons.lang.NotImplementedException;
import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.data.DenseBlock;
import org.tugraz.sysds.runtime.data.DenseBlockBool;
import org.tugraz.sysds.runtime.data.DenseBlockFactory;
import org.tugraz.sysds.runtime.data.DenseBlockLBool;
import org.tugraz.sysds.runtime.data.DenseBlockLFP32;
import org.tugraz.sysds.runtime.data.DenseBlockLFP64;
import org.tugraz.sysds.runtime.data.DenseBlockLString;
import org.tugraz.sysds.runtime.data.DenseBlockLInt32;
import org.tugraz.sysds.runtime.data.DenseBlockLInt64;

public class DenseBlockIncrementTest {
	@Test
	public void testIndexDenseBlock2FP32CountNonZero() {
		DenseBlock db = getDenseBlock2(ValueType.FP32);
		checkIncrement2(db);
	}

	@Test
	public void testIndexDenseBlock2FP64CountNonZero() {
		DenseBlock db = getDenseBlock2(ValueType.FP64);
		checkIncrement2(db);
	}

	@Test
	public void testIndexDenseBlock2BoolCountNonZero() {
		DenseBlock db = getDenseBlock2(ValueType.BOOLEAN);
		checkIncrement2(db);
	}

	@Test
	public void testIndexDenseBlock2Int32CountNonZero() {
		DenseBlock db = getDenseBlock2(ValueType.INT32);
		checkIncrement2(db);
	}

	@Test
	public void testIndexDenseBlock2Int64CountNonZero() {
		DenseBlock db = getDenseBlock2(ValueType.INT64);
		checkIncrement2(db);
	}

	@Test
	public void testIndexDenseBlock2StringCountNonZero() {
		DenseBlock db = getDenseBlock2(ValueType.STRING);
		try {
			checkIncrement2(db);
		} catch (UnsupportedOperationException ignored) {
		}
	}

	@Test
	public void testIndexDenseBlockLarge2FP32CountNonZero() {
		DenseBlock db = getDenseBlockLarge2(ValueType.FP32);
		checkIncrement2(db);
	}

	@Test
	public void testIndexDenseBlockLarge2FP64CountNonZero() {
		DenseBlock db = getDenseBlockLarge2(ValueType.FP64);
		checkIncrement2(db);
	}

	@Test
	public void testIndexDenseBlockLarge2BoolCountNonZero() {
		DenseBlock db = getDenseBlockLarge2(ValueType.BOOLEAN);
		checkIncrement2(db);
	}

	@Test
	public void testIndexDenseBlockLarge2Int32CountNonZero() {
		DenseBlock db = getDenseBlockLarge2(ValueType.INT32);
		checkIncrement2(db);
	}

	@Test
	public void testIndexDenseBlockLarge2Int64CountNonZero() {
		DenseBlock db = getDenseBlockLarge2(ValueType.INT64);
		checkIncrement2(db);
	}

	@Test
	public void testIndexDenseBlockLarge2StringCountNonZero() {
		DenseBlock db = getDenseBlockLarge2(ValueType.STRING);
		try {
			checkIncrement2(db);
		} catch (UnsupportedOperationException ignored) {
		}
	}

	private DenseBlock getDenseBlock2(ValueType vt) {
		return DenseBlockFactory.createDenseBlock(vt, new int[]{3, 5});
	}

	private DenseBlock getDenseBlockLarge2(ValueType vt) {
		int[] dims = {3, 5};
		switch (vt) {
			case FP32:
				return new DenseBlockLFP32(dims);
			case FP64:
				return new DenseBlockLFP64(dims);
			case BOOLEAN:
				return new DenseBlockLBool(dims);
			case INT32:
				return new DenseBlockLInt32(dims);
			case INT64:
				return new DenseBlockLInt64(dims);
			case STRING:
				return new DenseBlockLString(dims);
			default:
				throw new NotImplementedException();
		}
	}

	private void checkIncrement2(DenseBlock db) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 5; j++) {
				db.incr(i, j);
			}
		}
		Assert.assertEquals(3 * 5, db.countNonZeros());
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 5; j++) {
				Assert.assertEquals(1, db.get(i, j), 0);
			}
		}
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 5; j++) {
				db.incr(i, j, -1);
			}
		}
		if (db instanceof DenseBlockBool || db instanceof DenseBlockLBool) {
			Assert.assertEquals(3 * 5, db.countNonZeros());
		} else {
			Assert.assertEquals(0, db.countNonZeros());
		}

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 5; j++) {
				db.incr(i, j, 10);
			}
		}
		Assert.assertEquals(3 * 5, db.countNonZeros());
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 5; j++) {
				if (db instanceof DenseBlockBool || db instanceof DenseBlockLBool) {
					Assert.assertEquals(1, db.get(i, j), 0);
				} else {
					Assert.assertEquals(10, db.get(i, j), 0);
				}
			}
		}
	}
}
