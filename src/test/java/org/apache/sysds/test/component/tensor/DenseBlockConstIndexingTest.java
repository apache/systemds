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

import static org.junit.Assert.fail;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.data.*;
import org.apache.sysds.runtime.data.DenseBlockLBoolBitset;
import org.junit.Assert;
import org.junit.Test;

public class DenseBlockConstIndexingTest 
{
	@Test
	public void testIndexDenseBlock2FP32Const() {
		DenseBlock db = getDenseBlock2(ValueType.FP32);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				Assert.assertEquals(7.3, db.get(i, j), 1e-5);
	}
	
	@Test
	public void testIndexDenseBlock2FP64Const() {
		DenseBlock db = getDenseBlock2(ValueType.FP64);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				Assert.assertEquals(7.3, db.get(i, j), 0);
	}
	
	@Test
	public void testIndexDenseBlock2BoolConst() {
		DenseBlock db = getDenseBlock2(ValueType.BOOLEAN);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				Assert.assertEquals(1, db.get(i, j), 0);
	}

	@Test
	public void testIndexDenseBlock2TrueBoolConst() {
		DenseBlock db = new DenseBlockBoolArray(new int[] {3,5});
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++) {
				Assert.assertEquals(1, db.get(i, j), 0);
			}
	}
	
	@Test
	public void testIndexDenseBlock2Int32Const() {
		DenseBlock db = getDenseBlock2(ValueType.INT32);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				Assert.assertEquals(7, db.get(i, j), 0);
	}
	
	@Test
	public void testIndexDenseBlock2Int64Const() {
		DenseBlock db = getDenseBlock2(ValueType.INT64);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				Assert.assertEquals(7, db.get(i, j), 0);
	}

	@Test
	public void testIndexDenseBlock2StringConst() {
		try{

			DenseBlock db = getDenseBlock2(ValueType.STRING);
			db.set(new int[]{1,3}, "hello");
			Assert.assertEquals("hello", db.getString(new int[]{1,3}));
		}
		catch(Exception e){
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testIndexDenseBlockLarge2FP32Const() {
		DenseBlock db = getDenseBlockLarge2(ValueType.FP32);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				Assert.assertEquals(7.3, db.get(i, j), 1e-5);
	}

	@Test
	public void testIndexDenseBlockLarge2FP64Const() {
		DenseBlock db = getDenseBlockLarge2(ValueType.FP64);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				Assert.assertEquals(7.3, db.get(i, j), 0);
	}

	@Test
	public void testIndexDenseBlockLarge2BoolConst() {
		DenseBlock db = getDenseBlockLarge2(ValueType.BOOLEAN);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				Assert.assertEquals(1, db.get(i, j), 0);
	}



	@Test
	public void testIndexDenseBlockLarge2Int32Const() {
		DenseBlock db = getDenseBlockLarge2(ValueType.INT32);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				Assert.assertEquals(7, db.get(i, j), 0);
	}

	@Test
	public void testIndexDenseBlockLarge2Int64Const() {
		DenseBlock db = getDenseBlockLarge2(ValueType.INT64);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				Assert.assertEquals(7, db.get(i, j), 0);
	}

	@Test
	public void testIndexDenseBlockLarge2StringConst() {
		DenseBlock db = getDenseBlockLarge2(ValueType.STRING);
		db.set(new int[]{1,3}, "hello");
		Assert.assertEquals("hello", db.getString(new int[]{1,3}));
	}

	@Test
	public void testIndexDenseBlock3FP32Const() {
		DenseBlock db = getDenseBlock3(ValueType.FP32);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				for(int k=0; k<7; k++)
					Assert.assertEquals(7.3, db.get(new int[]{i,j,k}), 1e-5);
	}
	
	@Test
	public void testIndexDenseBlock3FP64Const() {
		DenseBlock db = getDenseBlock3(ValueType.FP64);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				for(int k=0; k<7; k++)
					Assert.assertEquals(7.3, db.get(new int[]{i,j,k}), 0);
	}
	
	@Test
	public void testIndexDenseBlock3BoolConst() {
		DenseBlock db = getDenseBlock3(ValueType.BOOLEAN);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				for(int k=0; k<7; k++)
					Assert.assertEquals(1, db.get(new int[]{i,j,k}), 0);
	}

	@Test
	public void testIndexDenseBlock3TrueBoolConst() {
		DenseBlock db = new DenseBlockBoolArray(new int[] {3,5,7});
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				for(int k=0; k<7; k++)
					Assert.assertEquals(1, db.get(new int[]{i,j,k}), 0);
	}
	
	@Test
	public void testIndexDenseBlock3Int32Const() {
		DenseBlock db = getDenseBlock3(ValueType.INT32);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				for(int k=0; k<7; k++)
					Assert.assertEquals(7, db.get(new int[]{i,j,k}), 0);
	}
	
	@Test
	public void testIndexDenseBlock3Int64Const() {
		DenseBlock db = getDenseBlock3(ValueType.INT64);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				for(int k=0; k<7; k++)
					Assert.assertEquals(7, db.get(new int[]{i,j,k}), 0);
	}

	@Test
	public void testIndexDenseBlock3StringConst() {
		DenseBlock db = getDenseBlock3(ValueType.STRING);
		db.set(new int[]{0,4,2}, "hello");
		Assert.assertEquals("hello", db.getString(new int[]{0,4,2}));
	}

	@Test
	public void testIndexDenseBlockLarge3FP32Const() {
		DenseBlock db = getDenseBlockLarge3(ValueType.FP32);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				for(int k=0; k<7; k++)
					Assert.assertEquals(7.3, db.get(new int[]{i,j,k}), 1e-5 );
	}

	@Test
	public void testIndexDenseBlockLarge3FP64Const() {
		DenseBlock db = getDenseBlockLarge3(ValueType.FP64);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				for(int k=0; k<7; k++)
					Assert.assertEquals(7.3, db.get(new int[]{i,j,k}), 0);
	}

	@Test
	public void testIndexDenseBlockLarge3BoolConst() {
		DenseBlock db = getDenseBlockLarge3(ValueType.BOOLEAN);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				for(int k=0; k<7; k++)
					Assert.assertEquals(1, db.get(new int[]{i,j,k}), 0);
	}

	@Test
	public void testIndexDenseBlockLarge3Int32Const() {
		DenseBlock db = getDenseBlockLarge3(ValueType.INT32);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				for(int k=0; k<7; k++)
					Assert.assertEquals(7, db.get(new int[]{i,j,k}), 0);
	}

	@Test
	public void testIndexDenseBlockLarge3Int64Const() {
		DenseBlock db = getDenseBlockLarge3(ValueType.INT64);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				for(int k=0; k<7; k++)
					Assert.assertEquals(7, db.get(new int[]{i,j,k}), 0);
	}

	@Test
	public void testIndexDenseBlockLarge3StringConst() {
		DenseBlock db = getDenseBlockLarge3(ValueType.STRING);
		db.set(new int[]{0,4,2}, "hello");
		Assert.assertEquals("hello", db.getString(new int[]{0,4,2}));
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
			case BOOLEAN: return new DenseBlockLBoolBitset(dims);
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
			case BOOLEAN: return new DenseBlockLBoolBitset(dims);
			case INT32: return new DenseBlockLInt32(dims);
			case INT64: return new DenseBlockLInt64(dims);
			case STRING: return new DenseBlockLString(dims);
			default: throw new NotImplementedException();
		}
	}
}
