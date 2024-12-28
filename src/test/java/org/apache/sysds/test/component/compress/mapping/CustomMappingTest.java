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

package org.apache.sysds.test.component.compress.mapping;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class CustomMappingTest {

	int[] data = new int[] {0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1,
		1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1,
		0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	@Test
	public void createBinary() {
		try {

			CompressedMatrixBlock.debug = true;
			MapToFactory.create(data, 2);
			MapToFactory.create(127, data, 2);
		}
		catch(RuntimeException e) {
			e.printStackTrace();
			throw e;
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void verifySpy() {
		CompressedMatrixBlock.debug = true;
		AMapToData d = MapToFactory.create(data, 2);
		AMapToData spy = spy(d);
		when(spy.getIndex(2)).thenReturn(32);
		assertThrows(DMLCompressionException.class, () -> spy.verify());
	}

	@Test
	public void equals() {
		CompressedMatrixBlock.debug = true;
		AMapToData d = MapToFactory.create(data, 2);
		AMapToData d2 = MapToFactory.create(data, 2);
		assertTrue(d.equals(d));
		assertTrue(d.equals(d2));
		assertFalse(d.equals(MapToFactory.create(new int[]{1,2,3}, 4)));
		assertFalse(d.equals(Integer.valueOf(23)));
	}

	@Test
	public void countRuns() {
		CompressedMatrixBlock.debug = true;
		AMapToData d = MapToFactory.create(new int[] {1, 1, 1, 1, 1, 2, 2, 2, 2, 2}, 3);
		AOffset o = OffsetFactory.createOffset(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
		assertEquals(d.countRuns(o), 2);
	}

	@Test
	public void countRuns2() {
		CompressedMatrixBlock.debug = true;
		AMapToData d = MapToFactory.create(new int[] {1, 1, 1, 1, 1, 2, 2, 2, 2, 2}, 3);
		AOffset o = OffsetFactory.createOffset(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11});
		assertEquals(d.countRuns(o), 3);
	}

	@Test
	public void getMax() {
		CompressedMatrixBlock.debug = true;
		AMapToData d = MapToFactory.create(new int[] {1, 1, 1, 1, 1, 2, 2, 2, 2, 2}, 3);
		assertEquals(d.getMax(), 2);
		d = MapToFactory.create(new int[] {1, 1, 1, 1, 1, 2, 2, 2, 5, 2}, 10);
		assertEquals(d.getMax(), 5);
		d = MapToFactory.create(new int[] {1, 1, 1, 9, 1, 2, 2, 2, 2, 2}, 10);
		assertEquals(d.getMax(), 9);
	}

	@Test
	public void copyInt(){
		CompressedMatrixBlock.debug = true;
		AMapToData d = MapToFactory.create(new int[] {10,9,8,7,6,5,4,3,2,1}, 11);
		AMapToData d2 = MapToFactory.create(new int[] {1,2,3,4,5,6,7,8,9,10}, Integer.MAX_VALUE -2);
		d.copy(d2);
		for(int i = 0; i < 10; i ++){
			assertEquals(d.getIndex(i), d2.getIndex(i));
		}
	}

	@Test
	public void setInteger(){
		CompressedMatrixBlock.debug = true;
		AMapToData d = MapToFactory.create(new int[] {10,9,8,7,6,5,4,3,2,1}, 11);
		
		for(int i = 0; i < 10; i ++){
			assertEquals(d.getIndex(i), 10- i);
		}
		d.set(4, Integer.valueOf(13));
		assertEquals(d.getIndex(4), 13);
	}

	@Test(expected = NotImplementedException.class)
	public void preAggDenseNonContiguous(){
		AMapToData d = MapToFactory.create(new int[] {10,9,8,7,6,5,4,3,2,1}, 11);
		MatrixBlock mb = new MatrixBlock();
		MatrixBlock spy = spy(mb);
		DenseBlock db = mock(DenseBlock.class);
		when(db.isContiguous()).thenReturn(false);
		when(spy.getDenseBlock()).thenReturn(db);

		d.preAggregateDense(spy, null, 10, 13,0, 10);
	}

}
