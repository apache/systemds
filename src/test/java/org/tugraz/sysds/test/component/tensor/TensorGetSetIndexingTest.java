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


public class TensorGetSetIndexingTest
{
	@Test
	public void testIndexTensorBlock2FP32SetGetCell() {
		TensorBlock tb = getTensorBlock2(ValueType.FP32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexTensorBlock2FP64SetGetCell() {
		TensorBlock tb = getTensorBlock2(ValueType.FP64);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexTensorBlock2BoolSetGetCell() {
		TensorBlock tb = getTensorBlock2(ValueType.BOOLEAN);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexTensorBlock2Int32SetGetCell() {
		TensorBlock tb = getTensorBlock2(ValueType.INT32);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexTensorBlock2Int64SetGetCell() {
		TensorBlock tb = getTensorBlock2(ValueType.INT64);
		checkSequence(setSequence(tb));
	}

	/* ToDo: Add constructor for Large Tensors?
	@Test
	public void testIndexTensorBlockLarge2FP32SetGetCell() {
		throw new NotImplementedException();
	}

	@Test
	public void testIndexTensorBlockLarge2FP64SetGetCell() {
		throw new NotImplementedException();
	}

	@Test
	public void testIndexTensorBlockLarge2BoolSetGetCell() {
		throw new NotImplementedException();
	}

	@Test
	public void testIndexTensorBlockLarge2Int32SetGetCell() {
		throw new NotImplementedException();
	}

	@Test
	public void testIndexTensorBlockLarge2Int64SetGetCell() {
	    throw new NotImplementedException();
	}*/

	@Test
	public void testIndexTensorBlock3FP32SetGetCell() {
		TensorBlock tb = getTensorBlock3(ValueType.FP32);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexTensorBlock3FP64SetGetCell() {
		TensorBlock tb = getTensorBlock3(ValueType.FP64);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexTensorBlock3BoolSetGetCell() {
		TensorBlock tb = getTensorBlock3(ValueType.BOOLEAN);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexTensorBlock3Int32SetGetCell() {
		TensorBlock tb = getTensorBlock3(ValueType.INT32);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexTensorBlock3Int64SetGetCell() {
		TensorBlock tb = getTensorBlock3(ValueType.INT64);
		checkSequence(setSequence(tb));
	}

	/* ToDo: Add constructor for Large Tensors?
	@Test
	public void testIndexTensorBlockLarge3FP32SetGetCell() {
		throw new NotImplementedException();
	}

	@Test
	public void testIndexTensorBlockLarge3FP64SetGetCell() {
		throw new NotImplementedException();
	}

	@Test
	public void testIndexTensorBlockLarge3BoolSetGetCell() {
		throw new NotImplementedException();
	}

	@Test
	public void testIndexTensorBlockLarge3Int32SetGetCell() {
		throw new NotImplementedException();
	}

	@Test
	public void testIndexTensorBlockLarge3Int64SetGetCell() {
	    throw new NotImplementedException();
	}*/

	private TensorBlock getTensorBlock2(ValueType vt) {
		// Todo: implement sparse for Tensor
		return new TensorBlock(vt, new int[] {3,5}, false);
	}
	
	private TensorBlock getTensorBlock3(ValueType vt) {
		// Todo: implement sparse for Tensor
		return new TensorBlock(vt, new int[] {3,5,7}, false);
	}

	private TensorBlock setSequence(TensorBlock tb) {
		if( tb.getNumDims() == 3 ) {
			int dim12 = 5*7;
			int dim1 =5, dim2 = 7;
			for(int i=0; i<tb.getNumRows(); i++)
				for(int j=0; j<dim1; j++)
					for(int k=0; k<dim2; k++)
						tb.set(new int[] {i,j,k}, (double)i*dim12+j*dim2+k);
		}
		else { //num dims = 2
			int dim1 = 5;
			for(int i=0; i<tb.getNumRows(); i++)
				for(int j=0; j<dim1; j++)
					tb.set(new int[]{i,j}, i*dim1+j);
		}
		return tb;
	}

	private void checkSequence(TensorBlock tb) {
		boolean isBool = tb.getValueType() == ValueType.BOOLEAN;
		if( tb.getNumDims() == 3 ) {
			int dim12 = 5*7;
			int dim1 = 5, dim2 = 7;
			for(int i=0; i<tb.getNumRows(); i++)
				for(int j=0; j<dim1; j++)
					for(int k=0; k<dim2; k++) {
						int val = i*dim12+j*dim2+k;
						double expected = isBool && val!=0 ? 1 : val;
						Assert.assertEquals(expected, tb.get(new int[] {i,j,k}), 0);
					}
		}
		else { //num dims = 2
			int dim1 = 5;
			for(int i=0; i<tb.getNumRows(); i++)
				for(int j=0; j<dim1; j++) {
					int val = i*dim1+j;
					double expected = isBool && val!=0 ? 1 : val;
					Assert.assertEquals(expected, tb.get(new int[]{i, j}), 0);
				}
		}
	}
}
