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

package org.tugraz.sysds.test.tensor;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.data.DenseBlock;
import org.tugraz.sysds.runtime.data.DenseBlockFactory;


public class DenseBlockConstIndexingTest 
{
	@Test
	public void testIndexDenseBlock2FP32Const() throws Exception {
		DenseBlock db = getDenseBlock2(ValueType.FP32);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				Assert.assertEquals(7.3, db.get(i, j), 1e-5);
	}
	
	@Test
	public void testIndexDenseBlock2FP64Const() throws Exception {
		DenseBlock db = getDenseBlock2(ValueType.FP64);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				Assert.assertEquals(7.3, db.get(i, j), 0);
	}
	
	@Test
	public void testIndexDenseBlock3FP32Const() throws Exception {
		DenseBlock db = getDenseBlock3(ValueType.FP32);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				for(int k=0; k<7; k++)
					Assert.assertEquals(7.3, db.get(new int[]{i,j,k}), 1e-5);
	}
	
	@Test
	public void testIndexDenseBlock3FP64Const() throws Exception {
		DenseBlock db = getDenseBlock3(ValueType.FP64);
		db.set(7.3);
		for(int i=0; i<db.numRows(); i++)
			for(int j=0; j<5; j++)
				for(int k=0; k<7; k++)
					Assert.assertEquals(7.3, db.get(new int[]{i,j,k}), 0);
	}
	
	private DenseBlock getDenseBlock2(ValueType vt) {
		return DenseBlockFactory.createDenseBlock(vt, new int[] {3,5});
	}
	
	private DenseBlock getDenseBlock3(ValueType vt) {
		return DenseBlockFactory.createDenseBlock(vt, new int[] {3,5,7});
	}
}
