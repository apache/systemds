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

package org.apache.sysds.test.component.compress.offset;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFactory;
import org.junit.Test;

public class OffsetPreAggTests {
	static{
		CompressedMatrixBlock.debug = true;
	}

	static DenseBlock db = DenseBlockFactory.createDenseBlock(2, 5);
	static{
		for(int i = 0; i < 5; i++) {
			db.set(0, i, i);
			db.set(1, i, i);
		}
	}

	@Test
	public void preAggDense() {
		try {

			AOffset off = OffsetFactory.createOffset(new int[] {0, 1, 2, 3, 4});
			DenseBlock db = DenseBlockFactory.createDenseBlock(1, 5);

			db.fill(1.0);

			for(MAP_TYPE t : MAP_TYPE.values()) {
				double[] preAV = new double[1];
				AMapToData data = MapToFactory.create(5, t);

				off.preAggregateDenseMap(db, preAV, 0, 1, 0, 5, 1, data);

				assertEquals(5.0, preAV[0], 0.0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void preAggDense2() {
		try {

			AOffset off = OffsetFactory.createOffset(new int[] {0, 1, 2, 3, 4});
		

			for(MAP_TYPE t : MAP_TYPE.values()) {
				double[] preAV = new double[1];
				AMapToData data = MapToFactory.create(5, t);

				off.preAggregateDenseMap(db, preAV, 0, 1, 0, 5, 1, data);

				assertEquals(10.0, preAV[0], 0.0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void preAggDense3() {
		try {

			AOffset off = OffsetFactory.createOffset(new int[] {0, 1, 2, 4});
		

			for(MAP_TYPE t : MAP_TYPE.values()) {
				double[] preAV = new double[1];
				AMapToData data = MapToFactory.create(5, t);

				off.preAggregateDenseMap(db, preAV, 0, 1, 0, 5, 1, data);

				assertEquals(7.0, preAV[0], 0.0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void preAggDense4() {
		try {

			AOffset off = OffsetFactory.createOffset(new int[] {0, 1, 2});
		
			for(MAP_TYPE t : MAP_TYPE.values()) {
				double[] preAV = new double[1];
				AMapToData data = MapToFactory.create(5, t);

				off.preAggregateDenseMap(db, preAV, 0, 1, 0, 5, 1, data);

				assertEquals(3.0, preAV[0], 0.0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void preAggDense5() {
		try {

			AOffset off = OffsetFactory.createOffset(new int[] {0, 2});
		

			for(MAP_TYPE t : MAP_TYPE.values()) {
				double[] preAV = new double[1];
				AMapToData data = MapToFactory.create(5, t);

				off.preAggregateDenseMap(db, preAV, 0, 1, 0, 5, 1, data);

				assertEquals(2.0, preAV[0], 0.0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void preAggDense6() {
		try {

			AOffset off = OffsetFactory.createOffset(new int[] {0, 2});
		
			for(MAP_TYPE t : MAP_TYPE.values()) {
				double[] preAV = new double[1];
				AMapToData data = MapToFactory.create(5, t);

				off.preAggregateDenseMap(db, preAV, 1, 2, 0, 5, 1, data);

				assertEquals(2.0, preAV[0], 0.0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void preAggDense7() {
		try {

			AOffset off = OffsetFactory.createOffset(new int[] {1, 2, 3});
		
			for(MAP_TYPE t : MAP_TYPE.values()) {
				double[] preAV = new double[1];
				AMapToData data = MapToFactory.create(5, t);

				off.preAggregateDenseMap(db, preAV, 1, 2, 4, 5, 1, data);

				assertEquals(0.0, preAV[0], 0.0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void preAggDense8() {
		try {

			AOffset off = OffsetFactory.createOffset(new int[] {3});
			

			for(MAP_TYPE t : MAP_TYPE.values()) {
				double[] preAV = new double[1];
				AMapToData data = MapToFactory.create(5, t);

				off.preAggregateDenseMap(db, preAV, 1, 2, 0, 2, 1, data);

				assertEquals(0.0, preAV[0], 0.0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void preAggDense9() {
		try {

			AOffset off = OffsetFactory.createOffset(new int[] {3});
		
			for(MAP_TYPE t : MAP_TYPE.values()) {
				double[] preAV = new double[1];
				AMapToData data = MapToFactory.create(5, t);

				off.preAggregateDenseMap(db, preAV, 1, 2, 0, 3, 1, data);

				assertEquals(0.0, preAV[0], 0.0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}


	@Test
	public void preAggDense10() {
		try {

			AOffset off = OffsetFactory.createOffset(new int[] {3,4,5});


			for(MAP_TYPE t : MAP_TYPE.values()) {
				double[] preAV = new double[1];
				AMapToData data = MapToFactory.create(5, t);

				off.preAggregateDenseMap(db, preAV, 1, 2, 0, 4, 1, data);

				assertEquals(3.0 , preAV[0], 0.0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void preAggDenseMultiRow1() {
		try {

			AOffset off = OffsetFactory.createOffset(new int[] {3});
		
			for(MAP_TYPE t : MAP_TYPE.values()) {
				double[] preAV = new double[2];
				AMapToData data = MapToFactory.create(5, t);

				off.preAggregateDenseMap(db, preAV, 0, 2, 0, 5, 1, data);

				assertEquals(3.0, preAV[0], 0.0);
				assertEquals(3.0, preAV[1], 0.0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void preAggDenseMultiRow2() {
		try {

			AOffset off = OffsetFactory.createOffset(new int[] {2, 3, 4});
		

			for(MAP_TYPE t : MAP_TYPE.values()) {
				double[] preAV = new double[2];
				AMapToData data = MapToFactory.create(5, t);

				off.preAggregateDenseMap(db, preAV, 0, 2, 0, 5, 1, data);

				assertEquals(2.0 + 3.0 + 4.0, preAV[0], 0.0);
				assertEquals(2.0 + 3.0 + 4.0, preAV[1], 0.0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void preAggDenseMultiRow3() {
		try {

			AOffset off = OffsetFactory.createOffset(new int[] {2, 3, 4});
		

			for(MAP_TYPE t : MAP_TYPE.values()) {
				double[] preAV = new double[2];
				AMapToData data = MapToFactory.create(5, t);

				off.preAggregateDenseMap(db, preAV, 0, 2, 0, 4, 1, data);

				assertEquals(2.0 + 3.0, preAV[0], 0.0);
				assertEquals(2.0 + 3.0, preAV[1], 0.0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test(expected = Exception.class)
	public void preAggDenseMultiRow4_nonContinuous() {

		AOffset off = OffsetFactory.createOffset(new int[] {2, 3, 4});
		
		DenseBlock spy = spy(db);
		when(spy.isContiguous()).thenReturn(false);

		double[] preAV = new double[2];
		AMapToData data = MapToFactory.create(5, MAP_TYPE.CHAR);

		off.preAggregateDenseMap(spy, preAV, 0, 2, 0, 4, 1, data);

		assertEquals(2.0 + 3.0, preAV[0], 0.0);
		assertEquals(2.0 + 3.0, preAV[1], 0.0);

	}
}
