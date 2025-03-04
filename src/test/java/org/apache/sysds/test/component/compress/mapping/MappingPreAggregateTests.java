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
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetByte;
import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class MappingPreAggregateTests {

	protected static final Log LOG = LogFactory.getLog(MappingPreAggregateTests.class.getName());

	public final int seed;
	public final MAP_TYPE type;
	public final int size;
	private final AMapToData m;
	private final AOffset o;
	private final MatrixBlock mb; // matrix block to preAggregate from.
	private final MatrixBlock sb; // Sparse block to preAggregate from.
	private final double[] preRef;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		for(MAP_TYPE t : new MAP_TYPE[] {MAP_TYPE.ZERO, MAP_TYPE.BIT, MAP_TYPE.UBYTE}) {
			tests.add(new Object[] {1, 1, t, 13});
			tests.add(new Object[] {3, 1, t, 13});
			tests.add(new Object[] {3, 1, t, 63});
			tests.add(new Object[] {3, 1, t, 64});
			tests.add(new Object[] {3, 1, t, 65});
			tests.add(new Object[] {5, 1, t, 1234});
			tests.add(new Object[] {5, 1, t, 13});
			tests.add(new Object[] {51, 1, t, 3241});
		}
		for(MAP_TYPE t : new MAP_TYPE[] {MAP_TYPE.BIT, MAP_TYPE.BYTE, MAP_TYPE.UBYTE, MAP_TYPE.CHAR, MAP_TYPE.INT}) {
			tests.add(new Object[] {1, 2, t, 13});
			tests.add(new Object[] {3, 2, t, 13});
			tests.add(new Object[] {3, 2, t, 63});
			tests.add(new Object[] {3, 2, t, 64});
			tests.add(new Object[] {3, 2, t, 65});
			tests.add(new Object[] {5, 2, t, 1234});
			tests.add(new Object[] {5, 2, t, 13});
			tests.add(new Object[] {51, 2, t, 3241});
		}

		for(MAP_TYPE t : new MAP_TYPE[] {MAP_TYPE.BYTE, MAP_TYPE.CHAR, MAP_TYPE.INT}) {
			tests.add(new Object[] {1, 10, t, 13});
			tests.add(new Object[] {3, 10, t, 13});
			tests.add(new Object[] {3, 10, t, 63});
			tests.add(new Object[] {3, 10, t, 64});
			tests.add(new Object[] {3, 10, t, 65});
			tests.add(new Object[] {5, 10, t, 1234});
			tests.add(new Object[] {5, 10, t, 13});
			tests.add(new Object[] {51, 10, t, 3241});
			tests.add(new Object[] {5, 180, t, 1000});
		}

		for(MAP_TYPE t : new MAP_TYPE[] {MAP_TYPE.CHAR, MAP_TYPE.CHAR_BYTE, MAP_TYPE.INT}) {
			tests.add(new Object[] {5, 300, t, 400});
			tests.add(new Object[] {5, 300, t, 1234});
			tests.add(new Object[] {51, 300, t, 3241});
		}
		return tests;
	}

	public MappingPreAggregateTests(int seed, int nUnique, MAP_TYPE type, int size) {
		CompressedMatrixBlock.debug = true;
		this.seed = seed;
		this.type = type;
		this.size = size;
		m = genBitMap(seed, nUnique, size, type);
		mb = TestUtils.generateTestMatrixBlock(5, size, 0, 100, 1.0, seed);
		sb = TestUtils.generateTestMatrixBlock(5, size, 0, 100, 0.2, seed);
		o = OneOffset.create(size);

		int nVal = m.getUnique();
		preRef = new double[nVal * mb.getNumRows()];
		m.preAggregateDense(mb, preRef, 0, mb.getNumRows(), 0, size);
	}

	protected static AMapToData genBitMap(int seed, int nUnique, int size, MAP_TYPE type) {
		final Random r = new Random(seed);
		AMapToData m = MapToFactory.create(size, nUnique);

		for(int i = 0; i < nUnique; i++) {
			m.set(i, i);
		}
		for(int i = nUnique; i < size; i++) {
			int v = r.nextInt(nUnique);
			m.set(i, v);
		}
		m = MapToFactory.resizeForce(m, type);
		return m;
	}

	@Test
	public void testPreAggRowZero() {
		testPreAggregateDenseSingleRow(0);
	}

	@Test
	public void testPreAggRowOne() {
		testPreAggregateDenseSingleRow(1);
	}

	@Test
	public void testPreAggRowTwo() {
		testPreAggregateDenseSingleRow(2);
	}

	public void testPreAggregateDenseSingleRow(int row) {
		try {
			final int size = m.size();
			double[] pre = new double[m.getUnique()];
			m.preAggregateDense(mb, pre, row, row + 1, 0, size);
			compareRes(preRef, pre, row);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.toString());
		}
	}

	@Test
	public void testPreAggSubPartOfRow01() {
		testPreAggSubSectionsRow(0, 3, size - 2);
	}

	@Test
	public void testPreAggSubPartOfRow02() {
		testPreAggSubSectionsRow(1, size / 2, size - 2);
	}

	@Test
	public void testPreAggSubPartOfRow03() {
		testPreAggSubSectionsRow(3, 1, size / 2);
	}

	public void testPreAggSubSectionsRow(int row, int cl, int cu) {
		try {
			final int size = m.size();
			double[] pre = new double[m.getUnique()];
			m.preAggregateDense(mb, pre, row, row + 1, 0, cl);
			m.preAggregateDense(mb, pre, row, row + 1, cl, cu);
			m.preAggregateDense(mb, pre, row, row + 1, cu, size);
			compareRes(preRef, pre, row);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.toString());
		}
	}

	@Test
	public void testPreAggRowsZeroAndOne() {
		testPreAggRowsRange(0, 2);
	}

	@Test
	public void testPreAggRowsOneAndTwo() {
		testPreAggRowsRange(1, 3);
	}

	@Test
	public void testPreAggRowsOneToFour() {
		testPreAggRowsRange(1, 4);
	}

	@Test
	public void testPreAggRowsThreeToFive() {
		testPreAggRowsRange(3, 5);
	}

	public void testPreAggRowsRange(int rl, int ru) {
		try {
			int nVal = m.getUnique();
			double[] pre = new double[nVal * (ru - rl)];
			m.preAggregateDense(mb, pre, rl, ru, 0, size);
			compareRes(preRef, pre, rl, ru);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.toString());
		}
	}

	@Test
	public void testPreAggRowsZeroAndOneCols() {
		testPreAggRowsColsRange(0, 2, 1, size - 1);
	}

	@Test
	public void testPreAggRowsOneAndTwoCols() {
		testPreAggRowsColsRange(1, 3, size / 2, size - 3);
	}

	@Test
	public void testPreAggRowsOneToFourCols() {
		testPreAggRowsColsRange(1, 4, 2, size / 2);
	}

	@Test
	public void testPreAggRowsThreeToFiveCols() {
		testPreAggRowsColsRange(3, 5, size - 2, size - 1);
	}

	public void testPreAggRowsColsRange(int rl, int ru, int cl, int cu) {
		try {
			int nVal = m.getUnique();
			double[] pre = new double[nVal * (ru - rl)];
			m.preAggregateDense(mb, pre, rl, ru, 0, cl);
			m.preAggregateDense(mb, pre, rl, ru, cl, cu);
			m.preAggregateDense(mb, pre, rl, ru, cu, size);
			compareRes(preRef, pre, rl, ru);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.toString());
		}
	}

	@Test
	public void testPreAggregateDenseSingleRowWithIndexes() {
		switch(type) {
			case INT:
				return;
			default:
				try {
					final int size = m.size();
					double[] pre = new double[m.getUnique()];
					m.preAggregateDense(mb.getDenseBlock(), pre, 0, 1, 0, size, o);
					compareRes(preRef, pre, 0);
				}
				catch(Exception e) {
					e.printStackTrace();
					fail(e.toString());
				}
		}
	}

	@Test
	public void testPreAggregateSparseSingleRowWithIndexes() {
		try {
			if(!sb.isInSparseFormat())
				return;
			double[] pre = new double[m.getUnique()];
			m.preAggregateSparse(sb.getSparseBlock(), pre, 0, 1, o);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.toString());
		}
	}

	@Test
	public void testPreAggregateSparseSingleRow() {
		try {
			if(!sb.isInSparseFormat())
				return;
			double[] pre = new double[m.getUnique()];
			m.preAggregateSparse(sb.getSparseBlock(), pre, 0, 1);
			verifyPreaggregate(m, sb, 0, 1, pre);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.toString());
		}
	}

	@Test
	public void testPreAggregateSparseMultiRow() {
		try {
			if(!sb.isInSparseFormat())
				return;
			double[] pre = new double[m.getUnique() * sb.getNumRows()];
			m.preAggregateSparse(sb.getSparseBlock(), pre, 0, sb.getNumRows());
			verifyPreaggregate(m, sb, 0, sb.getNumRows(), pre);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.toString());
		}
	}

	@Test
	public void testPreAggregateSparseEmptySingleRow() {
		try {
			if(!sb.isInSparseFormat())
				return;
			double[] pre = new double[m.getUnique()];
			MatrixBlock sb2 = new MatrixBlock(sb.getNumRows(), sb.getNumColumns(), 0,  SparseBlockFactory.createSparseBlock(sb.getNumRows()));
			m.preAggregateSparse(sb2.getSparseBlock(), pre, 0, 1);
			verifyPreaggregate(m, sb2, 0, 1, pre);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.toString());
		}
	}

	@Test
	public void testPreAggregateSparseEmptyMultiRow() {
		try {
			if(!sb.isInSparseFormat())
				return;
			double[] pre = new double[m.getUnique() * sb.getNumRows()];
			MatrixBlock sb2 = new MatrixBlock(sb.getNumRows(), sb.getNumColumns(), 0,  SparseBlockFactory.createSparseBlock(sb.getNumRows()));
			m.preAggregateSparse(sb2.getSparseBlock(), pre, 0, sb.getNumRows());
			verifyPreaggregate(m, sb2, 0, sb.getNumRows(), pre);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.toString());
		}
	}


	private void verifyPreaggregate(AMapToData m, MatrixBlock mb, int rl, int ru, double[] ret){

		double[] verification = new double[ret.length];
		for(int i = rl; i < ru; i++){
			for(int j = 0; j < mb.getNumColumns(); j++){
				verification[m.getIndex(j) + i * m.getUnique()] += mb.get(i,j);
			}
		}
		assertArrayEquals(verification, ret, 0);
	}


	private void compareRes(double[] expectedFull, double[] actual, int row) {
		String error = "\nNot equal elements with " + type + "  " + m.getUnique();
		int nVal = m.getUnique();
		for(int offE = row * nVal, offA = 0; offA < nVal; offE++, offA++)
			assertEquals(error, expectedFull[offE], actual[offA], 0.0001);

	}

	private void compareRes(double[] expectedFull, double[] actual, int rl, int ru) {
		int nVal = m.getUnique();
		for(int i = rl, offA = 0; i < ru; i++)
			for(int offE = i * nVal; offE < nVal * (i + 1); offE++, offA++)
				assertEquals(expectedFull[offE], actual[offA], 0.0001);
	}

	private static class OneOffset extends OffsetByte {
		private static final long serialVersionUID = 1910028460503867232L;

		private OneOffset(byte[] offsets, int offsetToFirst, int offsetToLast, int length) {
			super(offsets, offsetToFirst, offsetToLast, length);
		}

		protected static OneOffset create(int length) {
			int offsetToFirst = 0;
			int offsetToLast = length - 1;
			byte[] offsets = new byte[length - 1];
			for(int i = 0; i < offsets.length; i++)
				offsets[i] = 1;
			return new OneOffset(offsets, offsetToFirst, offsetToLast, length);
		}
	}
}
