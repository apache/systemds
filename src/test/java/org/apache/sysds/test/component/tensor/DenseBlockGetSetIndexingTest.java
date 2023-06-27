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

package org.apache.sysds.test.component.tensor;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.data.*;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ValueType;


public class DenseBlockGetSetIndexingTest
{
	@Test
	public void testIndexDenseBlock2FP32SetGetCell() {
		DenseBlock db = getDenseBlock2(ValueType.FP32);
		checkSequence(setSequence(db));
	}

	@Test
	public void testIndexDenseBlock2FP64SetGetCell() {
		DenseBlock db = getDenseBlock2(ValueType.FP64);
		checkSequence(setSequence(db));
	}
	
	@Test
	public void testIndexDenseBlock2BoolSetGetCell() {
		DenseBlock db = getDenseBlock2(ValueType.BITSET);
		checkSequence(setSequence(db));
	}

	@Test
	public void testIndexDenseBlock2TrueBoolSetGetCell() {
		DenseBlock db = getDenseBlock2(ValueType.BOOLEAN);
		checkSequence(setSequence(db));
	}
	
	@Test
	public void testIndexDenseBlock2Int32SetGetCell() {
		DenseBlock db = getDenseBlock2(ValueType.INT32);
		checkSequence(setSequence(db));
	}
	
	@Test
	public void testIndexDenseBlock2Int64SetGetCell() {
		DenseBlock db = getDenseBlock2(ValueType.INT64);
		checkSequence(setSequence(db));
	}

	@Test
	public void testIndexDenseBlock2StringSetGetCell() {
		DenseBlock db = getDenseBlock2(ValueType.STRING);
		checkSequence(setSequence(db));
	}

	@Test
	public void testIndexDenseBlockLarge2FP32SetGetCell() {
		DenseBlock db = getDenseBlockLarge2(ValueType.FP32);
		checkSequence(setSequence(db));
	}

	@Test
	public void testIndexDenseBlockLarge2FP64SetGetCell() {
		DenseBlock db = getDenseBlockLarge2(ValueType.FP64);
		checkSequence(setSequence(db));
	}

	@Test
	public void testIndexDenseBlockLarge2BoolSetGetCell() {
		DenseBlock db = getDenseBlockLarge2(ValueType.BITSET);
		checkSequence(setSequence(db));
	}

	@Test
	public void testIndexDenseBlockLarge2Int32SetGetCell() {
		DenseBlock db = getDenseBlockLarge2(ValueType.INT32);
		checkSequence(setSequence(db));
	}

	@Test
	public void testIndexDenseBlockLarge2Int64SetGetCell() {
		DenseBlock db = getDenseBlockLarge2(ValueType.INT64);
		checkSequence(setSequence(db));
	}

	@Test
	public void testIndexDenseBlockLarge2StringSetGetCell() {
		DenseBlock db = getDenseBlockLarge2(ValueType.STRING);
		checkSequence(setSequence(db));
	}

	@Test
	public void testIndexDenseBlock3FP32SetGetCell() {
		DenseBlock db = getDenseBlock3(ValueType.FP32);
		checkSequence(setSequence(db));
	}
	
	@Test
	public void testIndexDenseBlock3FP64SetGetCell() {
		DenseBlock db = getDenseBlock3(ValueType.FP64);
		checkSequence(setSequence(db));
	}
	
	@Test
	public void testIndexDenseBlock3BoolSetGetCell() {
		DenseBlock db = getDenseBlock3(ValueType.BITSET);
		checkSequence(setSequence(db));
	}

	@Test
	public void testIndexDenseBlock3TrueBoolSetGetCell() {
		DenseBlock db = getDenseBlock3(ValueType.BOOLEAN);
		checkSequence(setSequence(db));
	}
	
	@Test
	public void testIndexDenseBlock3Int32SetGetCell() {
		DenseBlock db = getDenseBlock3(ValueType.INT32);
		checkSequence(setSequence(db));
	}
	
	@Test
	public void testIndexDenseBlock3Int64SetGetCell() {
		DenseBlock db = getDenseBlock3(ValueType.INT64);
		checkSequence(setSequence(db));
	}

	@Test
	public void testIndexDenseBlock3StringSetGetCell() {
		DenseBlock db = getDenseBlock3(ValueType.STRING);
		checkSequence(setSequence(db));
	}

	@Test
	public void testIndexDenseBlockLarge3FP32SetGetCell() {
		DenseBlock db = getDenseBlockLarge3(ValueType.FP32);
		checkSequence(setSequence(db));
	}

	@Test
	public void testIndexDenseBlockLarge3FP64SetGetCell() {
		DenseBlock db = getDenseBlockLarge3(ValueType.FP64);
		checkSequence(setSequence(db));
	}

	@Test
	public void testIndexDenseBlockLarge3BoolSetGetCell() {
		DenseBlock db = getDenseBlockLarge3(ValueType.BITSET);
		checkSequence(setSequence(db));
	}

	@Test
	public void testIndexDenseBlockLarge3Int32SetGetCell() {
		DenseBlock db = getDenseBlockLarge3(ValueType.INT32);
		checkSequence(setSequence(db));
	}

	@Test
	public void testIndexDenseBlockLarge3Int64SetGetCell() {
		DenseBlock db = getDenseBlockLarge3(ValueType.INT64);
		checkSequence(setSequence(db));
	}

	@Test
	public void testIndexDenseBlockLarge3StringSetGetCell() {
		DenseBlock db = getDenseBlockLarge3(ValueType.STRING);
		checkSequence(setSequence(db));
	}

	private static DenseBlock getDenseBlock2(ValueType vt) {
		return DenseBlockFactory.createDenseBlock(vt, new int[] {3,5});
	}
	
	private static DenseBlock getDenseBlock3(ValueType vt) {
		return DenseBlockFactory.createDenseBlock(vt, new int[] {3,5,7});
	}

	private static DenseBlock getDenseBlockLarge2(ValueType vt) {
		int[] dims = {3,5};
		switch (vt) {
			case FP32: return new DenseBlockLFP32(dims);
			case FP64: return new DenseBlockLFP64(dims);
			case BITSET: return new DenseBlockLBoolBitset(dims);
			case INT32: return new DenseBlockLInt32(dims);
			case INT64: return new DenseBlockLInt64(dims);
			case STRING: return new DenseBlockLString(dims);
			default: throw new NotImplementedException();
		}
	}

	private static DenseBlock getDenseBlockLarge3(ValueType vt) {
		int[] dims = {3,5,7};
		switch (vt) {
			case FP32: return new DenseBlockLFP32(dims);
			case FP64: return new DenseBlockLFP64(dims);
			case BITSET: return new DenseBlockLBoolBitset(dims);
			case INT32: return new DenseBlockLInt32(dims);
			case INT64: return new DenseBlockLInt64(dims);
			case STRING: return new DenseBlockLString(dims);
			default: throw new NotImplementedException();
		}
	}

	private static DenseBlock setSequence(DenseBlock db) {
		if( db.numDims() == 3 ) {
			int dim12 = 5*7;
			int dim1 =5, dim2 = 7;
			for(int i=0; i<db.numRows(); i++)
				for(int j=0; j<dim1; j++)
					for(int k=0; k<dim2; k++)
						db.set(new int[]{i, j, k}, (double) i * dim12 + j * dim2 + k);
		}
		else { //num dims = 2
			int dim1 = 5;
			for(int i=0; i<db.numRows(); i++)
				for(int j=0; j<dim1; j++) {
					db.set(i, j, (double) i * dim1 + j);
				}
		}
		return db;
	}
	
	private static void checkSequence(DenseBlock db) {
		boolean isBool = (db instanceof DenseBlockBoolBitset) || (db instanceof DenseBlockLBoolBitset) || (db instanceof DenseBlockBoolArray);
		if( db.numDims() == 3 ) {
			int dim12 = 5*7;
			int dim1 = 5, dim2 = 7;
			for(int i=0; i<db.numRows(); i++)
				for(int j=0; j<dim1; j++)
					for(int k=0; k<dim2; k++) {
						int val = i*dim12+j*dim2+k;
						double expected = isBool && val != 0 ? 1 : val;
						Assert.assertEquals(expected, db.get(new int[]{i, j, k}), 0);
					}
		}
		else { //num dims = 2
			int dim1 = 5;
			for(int i=0; i<db.numRows(); i++)
				for(int j=0; j<dim1; j++) {
					int val = i*dim1+j;
					double expected = isBool && val != 0 ? 1 : val;
					Assert.assertEquals(expected, db.get(i, j), 0);
				}
		}
	}
}
