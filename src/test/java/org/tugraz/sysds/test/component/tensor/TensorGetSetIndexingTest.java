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
import org.tugraz.sysds.runtime.data.TensorBlock;
import org.tugraz.sysds.runtime.util.UtilFunctions;

import java.util.Arrays;


public class TensorGetSetIndexingTest
{
	private static int DIM0 = 3, DIM1 = 5, DIM2 = 7;
	// TODO large tensor tests
	@Test
	public void testIndexBasicTensor2FP32SetGetCell() {
		TensorBlock tb = getBasicTensor2(ValueType.FP32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexBasicTensor2FP64SetGetCell() {
		TensorBlock tb = getBasicTensor2(ValueType.FP64);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexBasicTensor2BoolSetGetCell() {
		TensorBlock tb = getBasicTensor2(ValueType.BOOLEAN);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexBasicTensor2Int32SetGetCell() {
		TensorBlock tb = getBasicTensor2(ValueType.INT32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexBasicTensor2Int64SetGetCell() {
		TensorBlock tb = getBasicTensor2(ValueType.INT64);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexBasicTensor3FP32SetGetCell() {
		TensorBlock tb = getBasicTensor3(ValueType.FP32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexBasicTensor3FP64SetGetCell() {
		TensorBlock tb = getBasicTensor3(ValueType.FP64);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexBasicTensor3BoolSetGetCell() {
		TensorBlock tb = getBasicTensor3(ValueType.BOOLEAN);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexBasicTensor3Int32SetGetCell() {
		TensorBlock tb = getBasicTensor3(ValueType.INT32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexBasicTensor3Int64SetGetCell() {
		TensorBlock tb = getBasicTensor3(ValueType.INT64);
		checkSequence(setSequence(tb));
	}

	private TensorBlock getBasicTensor2(ValueType vt) {
		// Todo: implement sparse for Tensor
		return new TensorBlock(vt, new int[] {DIM0,DIM1});
	}

	private TensorBlock getBasicTensor3(ValueType vt) {
		// Todo: implement sparse for Tensor
		return new TensorBlock(vt, new int[] {DIM0,DIM1,DIM2});
	}

	private TensorBlock setSequence(TensorBlock tb) {
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

	private void checkSequence(TensorBlock tb) {
		boolean isBool = (tb.isBasic() ? tb.getValueType() : tb.getSchema()[0]) == ValueType.BOOLEAN;
		if( tb.getNumDims() == DIM0 ) {
			int dim12 = DIM1 * DIM2;
			for(int i=0; i<tb.getNumRows(); i++)
				for(int j=0; j<DIM1; j++)
					for(int k=0; k<DIM2; k++) {
						int val = i*dim12+j*DIM2+k;
						double expected = isBool && val!=0 ? 1 : val;
						Object actualObj = tb.get(new int[]{i, j, k});
						ValueType vt = !tb.isBasic() ? tb.getSchema()[j] : tb.getValueType();
						double actual = UtilFunctions.objectToDouble(vt, actualObj);
						Assert.assertEquals(expected, actual, 0);
					}
		}
		else { //num dims = 2
			for(int i=0; i<tb.getNumRows(); i++)
				for(int j=0; j<DIM1; j++) {
					int val = i*DIM1+j;
					double expected = isBool && val!=0 ? 1 : val;
					ValueType vt = !tb.isBasic() ? tb.getSchema()[j] : tb.getValueType();
					double actual = UtilFunctions.objectToDouble(
						vt, tb.get(new int[]{i, j}));
					Assert.assertEquals(expected, actual, 0);
				}
		}
	}

	@Test
	public void testIndexDataTensor2FP32SetGetCell() {
		TensorBlock tb = getDataTensor2(ValueType.FP32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexDataTensor2FP64SetGetCell() {
		TensorBlock tb = getDataTensor2(ValueType.FP64);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexDataTensor2BoolSetGetCell() {
		TensorBlock tb = getDataTensor2(ValueType.BOOLEAN);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexDataTensor2Int32SetGetCell() {
		TensorBlock tb = getDataTensor2(ValueType.INT32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexDataTensor2Int64SetGetCell() {
		TensorBlock tb = getDataTensor2(ValueType.INT64);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexDataTensor3FP32SetGetCell() {
		TensorBlock tb = getDataTensor3(ValueType.FP32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexDataTensor3FP64SetGetCell() {
		TensorBlock tb = getDataTensor3(ValueType.FP64);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexDataTensor3BoolSetGetCell() {
		TensorBlock tb = getDataTensor3(ValueType.BOOLEAN);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexDataTensor3Int32SetGetCell() {
		TensorBlock tb = getDataTensor3(ValueType.INT32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexDataTensor3Int64SetGetCell() {
		TensorBlock tb = getDataTensor3(ValueType.INT64);
		checkSequence(setSequence(tb));
	}

	private TensorBlock getDataTensor2(ValueType vt) {
		ValueType[] schema = new ValueType[DIM1];
		Arrays.fill(schema, vt);
		return new TensorBlock(schema, new int[] {DIM0,DIM1});
	}

	private TensorBlock getDataTensor3(ValueType vt) {
		ValueType[] schema = new ValueType[DIM1];
		Arrays.fill(schema, vt);
		return new TensorBlock(schema, new int[] {DIM0,DIM1,DIM2});
	}
}
