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
import org.tugraz.sysds.runtime.data.DataTensor;
import org.tugraz.sysds.runtime.data.BasicTensor;


public class TensorGetSetIndexingTest
{
	private static int DIM0 = 3, DIM1 = 5, DIM2 = 7;
	// TODO large tensor tests
	@Test
	public void testIndexBasicTensor2FP32SetGetCell() {
		BasicTensor tb = getBasicTensor2(ValueType.FP32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexBasicTensor2FP64SetGetCell() {
		BasicTensor tb = getBasicTensor2(ValueType.FP64);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexBasicTensor2BoolSetGetCell() {
		BasicTensor tb = getBasicTensor2(ValueType.BOOLEAN);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexBasicTensor2Int32SetGetCell() {
		BasicTensor tb = getBasicTensor2(ValueType.INT32);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexBasicTensor2Int64SetGetCell() {
		BasicTensor tb = getBasicTensor2(ValueType.INT64);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexBasicTensor3FP32SetGetCell() {
		BasicTensor tb = getBasicTensor3(ValueType.FP32);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexBasicTensor3FP64SetGetCell() {
		BasicTensor tb = getBasicTensor3(ValueType.FP64);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexBasicTensor3BoolSetGetCell() {
		BasicTensor tb = getBasicTensor3(ValueType.BOOLEAN);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexBasicTensor3Int32SetGetCell() {
		BasicTensor tb = getBasicTensor3(ValueType.INT32);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexBasicTensor3Int64SetGetCell() {
		BasicTensor tb = getBasicTensor3(ValueType.INT64);
		checkSequence(setSequence(tb));
	}

	private BasicTensor getBasicTensor2(ValueType vt) {
		// Todo: implement sparse for Tensor
		return new BasicTensor(vt, new int[] {DIM0,DIM1}, false);
	}
	
	private BasicTensor getBasicTensor3(ValueType vt) {
		// Todo: implement sparse for Tensor
		return new BasicTensor(vt, new int[] {DIM0,DIM1,DIM2}, false);
	}

	private BasicTensor setSequence(BasicTensor tb) {
		if( tb.getNumDims() == DIM0 ) {
			int dim12 = DIM1*DIM2;
			for(int i=0; i<tb.getNumRows(); i++)
				for(int j=0; j<DIM1; j++)
					for(int k=0; k<DIM2; k++)
						tb.set(new int[] {i,j,k}, (double)i*dim12+j*DIM2+k);
		}
		else { //num dims = 2
			for(int i=0; i<tb.getNumRows(); i++)
				for(int j=0; j<DIM1; j++)
					tb.set(new int[]{i,j}, i*DIM1+j);
		}
		return tb;
	}

	private void checkSequence(BasicTensor tb) {
		boolean isBool = tb.getValueType() == ValueType.BOOLEAN;
		if( tb.getNumDims() == DIM0 ) {
			int dim12 = DIM1 * DIM2;
			for(int i=0; i<tb.getNumRows(); i++)
				for(int j=0; j<DIM1; j++)
					for(int k=0; k<DIM2; k++) {
						int val = i*dim12+j*DIM2+k;
						double expected = isBool && val!=0 ? 1 : val;
						Assert.assertEquals(expected, tb.get(new int[] {i,j,k}), 0);
					}
		}
		else { //num dims = 2
			for(int i=0; i<tb.getNumRows(); i++)
				for(int j=0; j<DIM1; j++) {
					int val = i*DIM1+j;
					double expected = isBool && val!=0 ? 1 : val;
					Assert.assertEquals(expected, tb.get(new int[]{i, j}), 0);
				}
		}
	}

	@Test
	public void testIndexDataTensor2FP32SetGetCell() {
		DataTensor tb = getDataTensor2(ValueType.FP32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexDataTensor2FP64SetGetCell() {
		DataTensor tb = getDataTensor2(ValueType.FP64);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexDataTensor2BoolSetGetCell() {
		DataTensor tb = getDataTensor2(ValueType.BOOLEAN);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexDataTensor2Int32SetGetCell() {
		DataTensor tb = getDataTensor2(ValueType.INT32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexDataTensor2Int64SetGetCell() {
		DataTensor tb = getDataTensor2(ValueType.INT64);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexDataTensor3FP32SetGetCell() {
		DataTensor tb = getDataTensor3(ValueType.FP32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexDataTensor3FP64SetGetCell() {
		DataTensor tb = getDataTensor3(ValueType.FP64);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexDataTensor3BoolSetGetCell() {
		DataTensor tb = getDataTensor3(ValueType.BOOLEAN);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexDataTensor3Int32SetGetCell() {
		DataTensor tb = getDataTensor3(ValueType.INT32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexDataTensor3Int64SetGetCell() {
		DataTensor tb = getDataTensor3(ValueType.INT64);
		checkSequence(setSequence(tb));
	}

	private DataTensor getDataTensor2(ValueType vt) {
		return new DataTensor(vt, new int[] {DIM0,DIM1});
	}

	private DataTensor getDataTensor3(ValueType vt) {
		return new DataTensor(vt, new int[] {DIM0,DIM1,DIM2});
	}

	private DataTensor setSequence(DataTensor tb) {
		if( tb.getNumDims() == DIM0 ) {
			int dim12 = DIM1*DIM2;
			for(int i=0; i<tb.getNumRows(); i++)
				for(int j=0; j<DIM1; j++)
					for(int k=0; k<DIM2; k++)
						tb.set(new int[] {i,j,k}, (double)i*dim12+j*DIM2+k);
		}
		else { //num dims = 2
			for(int i=0; i<tb.getNumRows(); i++)
				for(int j=0; j<DIM1; j++)
					tb.set(new int[]{i,j}, i*DIM1+j);
		}
		return tb;
	}

	private void checkSequence(DataTensor tb) {
		boolean isBool = tb.getSchema()[0] == ValueType.BOOLEAN;
		if( tb.getNumDims() == DIM0 ) {
			int dim12 = DIM1*DIM2;
			for(int i=0; i<tb.getNumRows(); i++)
				for(int j=0; j<DIM1; j++)
					for(int k=0; k<DIM2; k++) {
						int val = i*dim12+j*DIM2+k;
						double expected = isBool && val!=0 ? 1 : val;
						Assert.assertEquals(expected, tb.get(new int[] {i,j,k}), 0);
					}
		}
		else { //num dims = 2
			for(int i=0; i<tb.getNumRows(); i++)
				for(int j=0; j<DIM1; j++) {
					int val = i*DIM1+j;
					double expected = isBool && val!=0 ? 1 : val;
					Assert.assertEquals(expected, tb.get(new int[]{i, j}), 0);
				}
		}
	}
}
