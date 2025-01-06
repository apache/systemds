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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import java.util.Arrays;
import java.util.Random;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.IMapToDataGroup;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToBit;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToByte;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToChar;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToCharPByte;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToInt;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToUByte;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToZero;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFactory;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class CustomMappingTest {

	protected static final Log LOG = LogFactory.getLog(CustomMappingTest.class.getName());

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
		assertFalse(d.equals(MapToFactory.create(new int[] {1, 2, 3}, 4)));
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
	public void copyInt() {
		CompressedMatrixBlock.debug = true;
		AMapToData d = MapToFactory.create(new int[] {10, 9, 8, 7, 6, 5, 4, 3, 2, 1}, 11);
		AMapToData d2 = MapToFactory.create(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, Integer.MAX_VALUE - 2);
		d.copy(d2);
		for(int i = 0; i < 10; i++) {
			assertEquals(d.getIndex(i), d2.getIndex(i));
		}
	}

	@Test
	public void setInteger() {
		CompressedMatrixBlock.debug = true;
		AMapToData d = MapToFactory.create(new int[] {10, 9, 8, 7, 6, 5, 4, 3, 2, 1}, 11);

		for(int i = 0; i < 10; i++) {
			assertEquals(d.getIndex(i), 10 - i);
		}
		d.set(4, Integer.valueOf(13));
		assertEquals(d.getIndex(4), 13);
	}

	@Test(expected = NotImplementedException.class)
	public void preAggDenseNonContiguous() {
		AMapToData d = MapToFactory.create(new int[] {10, 9, 8, 7, 6, 5, 4, 3, 2, 1}, 11);
		MatrixBlock mb = new MatrixBlock();
		MatrixBlock spy = spy(mb);
		DenseBlock db = mock(DenseBlock.class);
		when(db.isContiguous()).thenReturn(false);
		when(spy.getDenseBlock()).thenReturn(db);

		d.preAggregateDense(spy, null, 10, 13, 0, 10);
	}

	@Test
	public void compareParallelCreate() throws Exception {
		int[] input = new int[100];
		Random r = new Random(32);
		for(int i = 0; i < 100; i++) {
			input[i] = r.nextInt(100);
		}

		AMapToData a = MapToFactory.create(101, input, 100, 10);
		AMapToData b = MapToFactory.create(101, input, 100);
		assertEquals(a, b);
	}

	@Test
	public void allocateMapToPByte() {
		assertEquals(MapToCharPByte.class, MapToFactory.create(10, MapToCharPByte.max).getClass());
	}

	@Test
	public void createSpecificType() {
		assertEquals(MapToCharPByte.class, MapToFactory.create(10, MAP_TYPE.CHAR_BYTE).getClass());
		assertEquals(MapToZero.class, MapToFactory.create(10, MAP_TYPE.ZERO).getClass());
		assertEquals(MapToBit.class, MapToFactory.create(10, MAP_TYPE.BIT).getClass());
		assertEquals(MapToByte.class, MapToFactory.create(10, MAP_TYPE.BYTE).getClass());
		assertEquals(MapToUByte.class, MapToFactory.create(10, MAP_TYPE.UBYTE).getClass());
		assertEquals(MapToChar.class, MapToFactory.create(10, MAP_TYPE.CHAR).getClass());
		assertEquals(MapToInt.class, MapToFactory.create(10, MAP_TYPE.INT).getClass());
	}

	@Test
	public void estimateInMemorySize() {
		for(int i = 0; i < 10; i++) {

			assertEquals(MapToFactory.estimateInMemorySize(i, i), MapToFactory.create(i, i).getInMemorySize());
			assertEquals(MapToFactory.estimateInMemorySize(i, 256), MapToFactory.create(i, 256).getInMemorySize());
			assertEquals(MapToFactory.estimateInMemorySize(i, 256 * 256),
				MapToFactory.create(i, 256 * 256).getInMemorySize());
			assertEquals(MapToFactory.estimateInMemorySize(i, 256 * 256 * 256),
				MapToFactory.create(i, 256 * 256 * 256).getInMemorySize());
			assertEquals(MapToFactory.estimateInMemorySize(i, 256 * 256 * 256 * 256),
				MapToFactory.create(i, 256 * 256 * 256 * 256).getInMemorySize());
			assertEquals(MapToFactory.estimateInMemorySize(i, MapToCharPByte.max),
				MapToFactory.create(i, MapToCharPByte.max).getInMemorySize());
			assertEquals(MapToFactory.estimateInMemorySize(i, Integer.MAX_VALUE),
				MapToFactory.create(i, Integer.MAX_VALUE).getInMemorySize());
		}
	}

	@Test
	public void createWithIntArrayList() {
		AMapToData a = MapToFactory.create(10, new IntArrayList(new int[] {1, 2, 3, 4}));
		for(int i = 0; i < 4; i++) {
			assertEquals(i + 1, a.getIndex(i));
		}
	}

	@Test
	public void resize() {
		int s = 10;
		for(MAP_TYPE m : MapToFactory.MAP_TYPE.values()) {
			AMapToData a = MapToFactory.create(s, m);
			a.getUpperBoundValue();
			for(int i = 1; i < Integer.MAX_VALUE / 2 && i < a.getUpperBoundValue(); i = i * 2) {
				for(int j = 0; j < s; j++) {
					a.set(j, (int) Math.max(0L, (long) i - j - 1));
				}
				AMapToData b = a.resize(i);
				String mm = a.toString() + " vs " + b.toString();
				for(int j = 0; j < s; j++) {
					assertEquals(mm, a.getIndex(j), b.getIndex(j));
				}
			}

		}
	}

	@Test
	public void resize2() {
		int s = 42;
		for(MAP_TYPE m : MapToFactory.MAP_TYPE.values()) {
			AMapToData a = MapToFactory.create(s, m);
			a.getUpperBoundValue();
			for(int i = 1; i < Integer.MAX_VALUE / 2 && i < a.getUpperBoundValue(); i = i * 2) {
				for(int j = 0; j < s; j++) {
					a.set(j, (int) Math.max(0L, (long) i - j - 1));
				}
				AMapToData b = a.resize(i);
				String mm = a.toString() + " vs " + b.toString();
				for(int j = 0; j < s; j++) {
					assertEquals(mm, a.getIndex(j), b.getIndex(j));
				}
			}

		}
	}

	@Test
	public void testBitSetFill() {
		AMapToData a = MapToFactory.create(100, MAP_TYPE.BIT);
		testFill(100, a);
	}

	@Test
	public void testBitSetFill64() {
		int length = 64 * 3;
		AMapToData a = MapToFactory.create(length, MAP_TYPE.BIT);
		testFill(length, a);
	}

	private void testFill(int length, AMapToData a) {
		for(int i = 0; i < length; i++) {
			assertEquals(0, a.getIndex(i));
		}

		a.fill(1);
		for(int i = 0; i < length; i++) {
			assertEquals(1, a.getIndex(i));
		}

		a.fill(0);
		for(int i = 0; i < length; i++) {
			assertEquals(0, a.getIndex(i));
		}
	}

	@Test
	public void testBitSetNextBitOutOfRange() {
		MapToBit m = new MapToBit(2, 100);
		assertEquals(-1, m.nextSetBit(0));
		assertEquals(-1, m.nextSetBit(1000));
		m.fill(1);
		assertEquals(-1, m.nextSetBit(1000));
		assertEquals(-1, m.nextSetBit(100));
		assertEquals(99, m.nextSetBit(99));
		assertEquals(98, m.nextSetBit(98));

	}

	@Test
	public void testBitSetNextBit() {
		MapToBit m = new MapToBit(2, 100);
		m.set(1,1);
		m.set(98,1);
		assertEquals(1, m.nextSetBit(0));
		assertEquals(98, m.nextSetBit(2));
		m.fill(1);
		for(int i = 0; i < 100; i++){
			m.set(i, 0);
		}

		assertEquals(-1, m.nextSetBit(0));
		

	}


	@Test
	public void decompressToRange() {
		double[] values = new double[] {1, 2, 3, 4, 5, 6};

		for(MAP_TYPE m : MapToFactory.MAP_TYPE.values()) {
			AMapToData map = MapToFactory.create(new int[] {0, 1, 2, 3, 4, 5, 1, 2, 3}, m);
			int rl = 0;
			int ru = map.size();
			int off = 0;

			evalDecompressRange(values, map, rl, ru, off);
		}
	}

	@Test
	public void decompressToRange2() {
		double[] values = new double[] {1, 2, 3, 4, 5, 6.3};

		for(MAP_TYPE m : MapToFactory.MAP_TYPE.values()) {
			AMapToData map = MapToFactory.create(new int[] {0, 1, 2, 3, 4, 5, 1, 2, 3}, m);
			int rl = 3;
			int ru = map.size();
			int off = 0;

			evalDecompressRange(values, map, rl, ru, off);
		}
	}

	@Test
	public void decompressToRange3() {
		double[] values = new double[] {1, 2, 3, 4, 5, 6.3};

		for(MAP_TYPE m : MapToFactory.MAP_TYPE.values()) {
			AMapToData map = MapToFactory.create(new int[] {0, 1, 2, 3, 4, 5, 1, 2, 3}, m);
			int rl = 1;
			int ru = map.size() - 2;
			int off = 0;

			evalDecompressRange(values, map, rl, ru, off);
		}
	}

	@Test
	public void decompressToRangeOffset1() {
		double[] values = new double[] {1, 2, 3, 4, 5, 6.3};

		for(MAP_TYPE m : MapToFactory.MAP_TYPE.values()) {
			AMapToData map = MapToFactory.create(new int[] {0, 1, 2, 3, 4, 5, 1, 2, 3}, m);
			int rl = 1;
			int ru = map.size() - 2;
			int off = -1;

			evalDecompressRange(values, map, rl, ru, off);
		}
	}

	@Test
	public void decompressToRangeOffset2() {
		double[] values = new double[] {1, 2, 3, 4, 5, 6.3};

		for(MAP_TYPE m : MapToFactory.MAP_TYPE.values()) {
			AMapToData map = MapToFactory.create(new int[] {0, 1, 2, 3, 4, 5, 1, 2, 3}, m);
			int rl = 1;
			int ru = map.size() - 2;
			int off = 1;

			evalDecompressRange(values, map, rl, ru, off);
		}
	}

	private void evalDecompressRange(double[] values, AMapToData map, int rl, int ru, int off) {
		double[] ret = new double[map.size() + off];
		map.decompressToRange(ret, rl, ru, off, values);
		String r = Arrays.toString(ret);
		for(int i = 0; i < ret.length; i++) {
			if(i < rl + off || i >= ru + off)
				assertEquals(r + "index : " + i, 0, ret[i], 0);
			else
				assertEquals(r + "index : " + i, values[map.getIndex(i - off)], ret[i], 0);
		}
	}

	@Test
	public void isEmptyBitSet() {
		MapToBit m = new MapToBit(2, 1000);
		assertTrue(m.isEmpty());
		m.set(134, 1);
		assertFalse(m.isEmpty());
		m.set(134, 0);
		assertTrue(m.isEmpty());
		m.fill(1);
		assertFalse(m.isEmpty());
		m.fill(0);
		assertTrue(m.isEmpty());
	}

	@Test(expected = RuntimeException.class)
	public void appendNonZero() {
		MapToZero m = new MapToZero(10);
		IMapToDataGroup g = mock(IMapToDataGroup.class);
		when(g.getMapToData()).thenReturn(new MapToBit(2, 10));
		m.appendN(new IMapToDataGroup[] {g});
	}

	@Test
	public void getType() {

		for(MAP_TYPE m : MapToFactory.MAP_TYPE.values()) {
			assertEquals(m, MapToFactory.create(10, m).getType());
		}
	}

	@Test
	public void setAndGet() {
		Random r = new Random(324);
		for(MAP_TYPE m : MapToFactory.MAP_TYPE.values()) {
			AMapToData mm = MapToFactory.create(10, m);
			int v = MapToFactory.getMaxPossible(m);

			assertEquals(v, mm.setAndGet(1, v));
			if(v != 0){
				for(int i = 0; i < 100; i++){
					int rv = r.nextInt(v);
					int ri = r.nextInt(mm.size());
					assertEquals(rv, mm.setAndGet(ri, rv));
				}
			}
		}
	}


	@Test 
	public void nothingTestsForMapToZero(){
		MapToZero m = new MapToZero(10);
		m.copyBit(null); // do nothing
		m.replace(-1,10);// do nothing
		m.set(1,1,1,null); // do nothing
		assertEquals(0, m.getUpperBoundValue());
		assertEquals(m, new MapToZero(10));
		
	}

	@Test 
	public void mapToZeroSlice(){
		MapToZero m = new MapToZero(10);
		AMapToData m2 = m.slice(3,8); // return new.
		assertEquals(new MapToZero(8-3),m2);
		
	}


	@Test 
	public void mapToBitEmptySlice(){
		MapToBit m = new MapToBit(2, 20);
		AMapToData m2 = m.slice(3,8); // return new.
		assertEquals(new MapToZero(8-3),m2);
		
	}

	@Test 
	public void mapToZeroEquals(){
		MapToZero m = new MapToZero(10);
		assertNotEquals(MapToFactory.create(10, MAP_TYPE.BYTE),m);
		
	}


	@Test 
	public void sparseMM(){
		for(MAP_TYPE t : MAP_TYPE.values()){
			if(t == MAP_TYPE.ZERO)
				continue;
			AMapToData map = MapToFactory.create(new int[] {0,1,1}, t);
			SparseBlock sb = SparseBlockFactory.createIdentityMatrix(3);
			DenseBlock ret = DenseBlockFactory.createDenseBlock(new double[3 * 10], 3, 10);
			IDictionary dict = Dictionary.create(new double[]{1,1,1,2,2,2,3,3});
			IColIndex cols = ColIndexFactory.create(new int[]{1,4,8});
	
			map.lmSparseMatrixRow(sb, 0, ret, cols, dict);
	
			for(int i = 0; i < cols.size(); i++){
				assertEquals(1, ret.get(0, cols.get(i)), 0);
			}
	
			map.lmSparseMatrixRow(sb, 1, ret, cols, dict);
			for(int i = 0; i < cols.size(); i++){
				assertEquals(2, ret.get(1, cols.get(i)), 0);
			}
	
			map.lmSparseMatrixRow(SparseBlockFactory.createSparseBlock(10), 1, ret, cols, dict);
			for(int i = 0; i < cols.size(); i++){
				assertEquals(2, ret.get(1, cols.get(i)), 0);
			}
	
			for(int i = 0; i < 10; i++){
				assertEquals(0, ret.get(2, i), 0);
			}
	
			assertEquals(6, ret.countNonZeros());
		}

	}


	@Test 
	public void counts(){
		MapToBit m = new MapToBit(2, 134);
		m.set(3,1);
		m.set(10,1);
		m.set(110,1);
		int[] counts = m.getCounts();
		for(MAP_TYPE t : MAP_TYPE.values()){
			if(t == MAP_TYPE.ZERO)
				continue;
			AMapToData d = MapToFactory.resizeForce(m, t);
			assertArrayEquals(counts, d.getCounts());
		}


	}

	@Test 
	public void countRunsAlternative(){
		MapToBit m = new MapToBit(2, 134);
		m.set(3,1);
		m.set(10,1);
		m.set(110,1);
		int counts = m.countRuns();
		for(MAP_TYPE t : MAP_TYPE.values()){
			if(t == MAP_TYPE.ZERO)
				continue;
			AMapToData d = MapToFactory.resizeForce(m, t);
			assertEquals(counts, d.countRuns());
		}


	}


	@Test 
	public void bitToZero(){
		MapToBit m = new MapToBit(2, 10);
		m.fill(1);
		AMapToData a = m.resize(1);
		for(int i = 0; i < 10; i++)
			assertEquals(0, a.getIndex(i));
	}

	@Test(expected = RuntimeException.class) 
	public void invalidJoin(){
		MapToBit a = new MapToBit(2, 100);
		MapToBit b = new MapToBit(2, 200);

		a.preAggregateDDC_DDCSingleCol(b, null,null);
	}

	@Test 
	public void equalsMapToZZero(){
		MapToZero m = new MapToZero(10);
		assertEquals(m, new MapToZero(10));
		assertNotEquals(m, new MapToZero(11));
		assertNotEquals(m, new MapToZero(1));
		assertNotEquals(m, new MapToBit(2, 1));
		assertNotEquals(m, new MapToBit(2, 10));
	}

}
