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

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.data.DenseBlock;
import org.tugraz.sysds.runtime.data.DenseBlockBool;
import org.tugraz.sysds.runtime.data.DenseBlockFactory;


public class DenseBlockGetSetIndexing 
{
	@Test
	public void testIndexDenseBlock2FP32Const() throws Exception {
		DenseBlock db = getDenseBlock2(ValueType.FP32);
		checkSequence(setSequence(db));
	}
	
	@Test
	public void testIndexDenseBlock2FP64Const() throws Exception {
		DenseBlock db = getDenseBlock2(ValueType.FP64);
		checkSequence(setSequence(db));
	}
	
	@Test
	public void testIndexDenseBlock2BoolConst() throws Exception {
		DenseBlock db = getDenseBlock2(ValueType.BOOLEAN);
		checkSequence(setSequence(db));
	}
	
	@Test
	public void testIndexDenseBlock2Int32Const() throws Exception {
		DenseBlock db = getDenseBlock2(ValueType.INT32);
		checkSequence(setSequence(db));
	}
	
	@Test
	public void testIndexDenseBlock2Int64Const() throws Exception {
		DenseBlock db = getDenseBlock2(ValueType.INT64);
		checkSequence(setSequence(db));
	}
	
	@Test
	public void testIndexDenseBlock3FP32Const() throws Exception {
		DenseBlock db = getDenseBlock3(ValueType.FP32);
		checkSequence(setSequence(db));
	}
	
	@Test
	public void testIndexDenseBlock3FP64Const() throws Exception {
		DenseBlock db = getDenseBlock3(ValueType.FP64);
		checkSequence(setSequence(db));
	}
	
	@Test
	public void testIndexDenseBlock3BoolConst() throws Exception {
		DenseBlock db = getDenseBlock3(ValueType.BOOLEAN);
		checkSequence(setSequence(db));
	}
	
	@Test
	public void testIndexDenseBlock3Int32Const() throws Exception {
		DenseBlock db = getDenseBlock3(ValueType.INT32);
		checkSequence(setSequence(db));
	}
	
	@Test
	public void testIndexDenseBlock3Int64Const() throws Exception {
		DenseBlock db = getDenseBlock3(ValueType.INT64);
		checkSequence(setSequence(db));
	}
	
	private DenseBlock getDenseBlock2(ValueType vt) {
		return DenseBlockFactory.createDenseBlock(vt, new int[] {3,5});
	}
	
	private DenseBlock getDenseBlock3(ValueType vt) {
		return DenseBlockFactory.createDenseBlock(vt, new int[] {3,5,7});
	}
	
	private DenseBlock setSequence(DenseBlock db) {
		if( db.numDims() == 3 ) {
			int dim12 = 5*7;
			int dim1 =5, dim2 = 7;
			for(int i=0; i<db.numRows(); i++)
				for(int j=0; j<dim1; j++)
					for(int k=0; k<dim2; k++)
						db.set(new int[] {i,j,k}, (double)i*dim12+j*dim2+k);
		}
		else { //num dims = 2
			int dim1 = 5;
			for(int i=0; i<db.numRows(); i++)
				for(int j=0; j<dim1; j++)
					db.set(i, j, i*dim1+j);
		}
		return db;
	}
	
	private void checkSequence(DenseBlock db) {
		boolean isBool = db instanceof DenseBlockBool;
		if( db.numDims() == 3 ) {
			int dim12 = 5*7;
			int dim1 = 5, dim2 = 7;
			for(int i=0; i<db.numRows(); i++)
				for(int j=0; j<dim1; j++)
					for(int k=0; k<dim2; k++) {
						int val = i*dim12+j*dim2+k;
						double expected = isBool && val!=0 ? 1 : val;
						Assert.assertEquals(db.get(new int[] {i,j,k}), expected, 0);
					}
		}
		else { //num dims = 2
			int dim1 = 5;
			for(int i=0; i<db.numRows(); i++)
				for(int j=0; j<dim1; j++) {
					int val = i*dim1+j;
					double expected = isBool && val!=0 ? 1 : val;
					Assert.assertEquals(db.get(i, j), expected, 0);
				}
		}
	}
}
