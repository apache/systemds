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
import org.tugraz.sysds.runtime.data.HeterogTensor;
import org.tugraz.sysds.runtime.data.HomogTensor;


public class TensorGetSetIndexingTest
{
	private static int DIM0 = 3, DIM1 = 5, DIM2 = 7;
	// TODO large tensor tests
	@Test
	public void testIndexHomogTensor2FP32SetGetCell() {
		HomogTensor tb = getHomogTensor2(ValueType.FP32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexHomogTensor2FP64SetGetCell() {
		HomogTensor tb = getHomogTensor2(ValueType.FP64);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexHomogTensor2BoolSetGetCell() {
		HomogTensor tb = getHomogTensor2(ValueType.BOOLEAN);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexHomogTensor2Int32SetGetCell() {
		HomogTensor tb = getHomogTensor2(ValueType.INT32);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexHomogTensor2Int64SetGetCell() {
		HomogTensor tb = getHomogTensor2(ValueType.INT64);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexHomogTensor3FP32SetGetCell() {
		HomogTensor tb = getHomogTensor3(ValueType.FP32);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexHomogTensor3FP64SetGetCell() {
		HomogTensor tb = getHomogTensor3(ValueType.FP64);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexHomogTensor3BoolSetGetCell() {
		HomogTensor tb = getHomogTensor3(ValueType.BOOLEAN);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexHomogTensor3Int32SetGetCell() {
		HomogTensor tb = getHomogTensor3(ValueType.INT32);
		checkSequence(setSequence(tb));
	}
	
	@Test
	public void testIndexHomogTensor3Int64SetGetCell() {
		HomogTensor tb = getHomogTensor3(ValueType.INT64);
		checkSequence(setSequence(tb));
	}

	private HomogTensor getHomogTensor2(ValueType vt) {
		// Todo: implement sparse for Tensor
		return new HomogTensor(vt, new int[] {DIM0,DIM1}, false);
	}
	
	private HomogTensor getHomogTensor3(ValueType vt) {
		// Todo: implement sparse for Tensor
		return new HomogTensor(vt, new int[] {DIM0,DIM1,DIM2}, false);
	}

	private HomogTensor setSequence(HomogTensor tb) {
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

	private void checkSequence(HomogTensor tb) {
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
	public void testIndexHeterogTensor2FP32SetGetCell() {
		HeterogTensor tb = getHeterogTensor2(ValueType.FP32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexHeterogTensor2FP64SetGetCell() {
		HeterogTensor tb = getHeterogTensor2(ValueType.FP64);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexHeterogTensor2BoolSetGetCell() {
		HeterogTensor tb = getHeterogTensor2(ValueType.BOOLEAN);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexHeterogTensor2Int32SetGetCell() {
		HeterogTensor tb = getHeterogTensor2(ValueType.INT32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexHeterogTensor2Int64SetGetCell() {
		HeterogTensor tb = getHeterogTensor2(ValueType.INT64);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexHeterogTensor3FP32SetGetCell() {
		HeterogTensor tb = getHeterogTensor3(ValueType.FP32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexHeterogTensor3FP64SetGetCell() {
		HeterogTensor tb = getHeterogTensor3(ValueType.FP64);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexHeterogTensor3BoolSetGetCell() {
		HeterogTensor tb = getHeterogTensor3(ValueType.BOOLEAN);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexHeterogTensor3Int32SetGetCell() {
		HeterogTensor tb = getHeterogTensor3(ValueType.INT32);
		checkSequence(setSequence(tb));
	}

	@Test
	public void testIndexHeterogTensor3Int64SetGetCell() {
		HeterogTensor tb = getHeterogTensor3(ValueType.INT64);
		checkSequence(setSequence(tb));
	}

	private HeterogTensor getHeterogTensor2(ValueType vt) {
		return new HeterogTensor(vt, new int[] {DIM0,DIM1});
	}

	private HeterogTensor getHeterogTensor3(ValueType vt) {
		return new HeterogTensor(vt, new int[] {DIM0,DIM1,DIM2});
	}

	private HeterogTensor setSequence(HeterogTensor tb) {
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

	private void checkSequence(HeterogTensor tb) {
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
