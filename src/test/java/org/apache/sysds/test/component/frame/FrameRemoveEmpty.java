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

package org.apache.sysds.test.component.frame;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class FrameRemoveEmpty {
	protected static final Log LOG = LogFactory.getLog(FrameRemoveEmpty.class.getName());

	@Test
	public void removeEmptyCols() {
		FrameBlock in = getTestCase0();
		FrameBlock out = in.removeEmptyOperations(false, false, null);
		assertEquals(out.getNumColumns(), 0);
	}

	@Test
	public void removeEmptyColsWithNames() {
		FrameBlock in = getTestCase1();
		in.setColumnName(0, "Special Column");
		FrameBlock out = in.removeEmptyOperations(false, false, null);
		assertEquals(1, out.getNumColumns());
		assertEquals("Special Column", out.getColumnName(0));
	}

	@Test
	public void removeEmptyColsNoEmptyReturn() {
		FrameBlock in = getTestCase1();
		FrameBlock out = in.removeEmptyOperations(false, true, null);
		assertEquals(1, out.getNumColumns());
		assertEquals(null, out.get(0, 0));
		assertEquals(null, out.get(1, 0));
	}

	@Test
	public void removeEmptyColsNotEmpty() {
		FrameBlock in = getTestCase1();
		FrameBlock out = in.removeEmptyOperations(false, false, null);
		assertEquals(1, out.getNumColumns());
		assertEquals(null, out.get(0, 0));
		assertEquals(null, out.get(1, 0));
	}

	@Test
	public void removeEmptyColsWithSelect() {
		FrameBlock in = getTestCase2();
		MatrixBlock select = new MatrixBlock(1, 2, new double[] {0, 1});
		select.recomputeNonZeros();
		FrameBlock out = in.removeEmptyOperations(false, false, select);
		assertEquals(1, out.getNumColumns());
		assertEquals("You", out.get(4, 0));
	}

	@Test
	public void removeEmptyColsWithSelectAll() {
		FrameBlock in = getTestCase2();
		MatrixBlock select = new MatrixBlock(1, 2, new double[] {1, 1});
		select.recomputeNonZeros();
		FrameBlock out = in.removeEmptyOperations(false, false, select);
		assertEquals(2, out.getNumColumns());
		assertEquals("Hi", out.get(3, 0));
		assertEquals("You", out.get(4, 1));
	}

	@Test
	public void removeEmptyColsWithSelectAllNamedCol() {
		FrameBlock in = getTestCase2();
		in.setColumnName(1, "MEEE");
		MatrixBlock select = new MatrixBlock(1, 2, new double[] {1, 1});
		select.recomputeNonZeros();
		FrameBlock out = in.removeEmptyOperations(false, false, select);
		assertEquals(2, out.getNumColumns());
		assertEquals("Hi", out.get(3, 0));
		assertEquals("You", out.get(4, 1));
		assertEquals("MEEE", out.getColumnName(1));
	}

	@Test
	public void removeEmptyColsWithSelectNamedCol() {
		FrameBlock in = getTestCase2();
		in.setColumnName(1, "MEEE");
		MatrixBlock select = new MatrixBlock(1, 2, new double[] {0, 1});
		select.recomputeNonZeros();
		FrameBlock out = in.removeEmptyOperations(false, false, select);
		assertEquals(1, out.getNumColumns());
		assertEquals("You", out.get(4, 0));
		assertEquals("MEEE", out.getColumnName(0));
	}

	@Test
	public void removeEmptyColsWithSelectNamedColEmptyReturn() {
		FrameBlock in = getTestCase2();
		in.setColumnName(1, "MEEE");
		MatrixBlock select = new MatrixBlock(1, 2, new double[] {0, 0});
		select.recomputeNonZeros();
		FrameBlock out = in.removeEmptyOperations(false, true, select);
		assertEquals(1, out.getNumColumns());
		assertEquals("C1", out.getColumnName(0));
	}

	@Test(expected = DMLRuntimeException.class)
	public void removeEmptyInvalidCols_1() {
		FrameBlock in = getTestCase2();
		MatrixBlock select = new MatrixBlock(1, 3, false);
		select.setNonZeros(0);
		in.removeEmptyOperations(false, true, select);
	}

	@Test(expected = DMLRuntimeException.class)
	public void removeEmptyInvalidCols_2() {
		FrameBlock in = getTestCase2();
		MatrixBlock select = new MatrixBlock(1, 1, false);
		select.setNonZeros(0);
		in.removeEmptyOperations(false, true, select);
	}

	@Test(expected = DMLRuntimeException.class)
	public void removeEmptyInvalidCols_3() {
		FrameBlock in = getTestCase2();
		MatrixBlock select = new MatrixBlock(2, 1, false);
		select.setNonZeros(0);
		in.removeEmptyOperations(false, true, select);
	}

	@Test(expected = DMLRuntimeException.class)
	public void removeEmptyInvalidCols_4() {
		FrameBlock in = getTestCase2();
		MatrixBlock select = new MatrixBlock(2, 2, false);
		select.setNonZeros(0);
		in.removeEmptyOperations(false, true, select);
	}

	@Test(expected = DMLRuntimeException.class)
	public void removeEmptyInvalidRows_1() {
		FrameBlock in = getTestCase2();
		MatrixBlock select = new MatrixBlock(2, 2, false);
		select.setNonZeros(0);
		in.removeEmptyOperations(true, true, select);
	}

	@Test(expected = DMLRuntimeException.class)
	public void removeEmptyInvalidRows_2() {
		FrameBlock in = getTestCase2();
		MatrixBlock select = new MatrixBlock(10, 2, false);
		select.setNonZeros(0);
		in.removeEmptyOperations(true, true, select);
	}

	@Test(expected = DMLRuntimeException.class)
	public void removeEmptyInvalidRows_3() {
		FrameBlock in = getTestCase2();
		MatrixBlock select = new MatrixBlock(1, 10, false);
		select.setNonZeros(0);
		in.removeEmptyOperations(true, true, select);
	}

	@Test(expected = DMLRuntimeException.class)
	public void removeEmptyInvalidRows_4() {
		FrameBlock in = getTestCase2();
		MatrixBlock select = new MatrixBlock(2, 10, false);
		select.setNonZeros(0);
		in.removeEmptyOperations(true, true, select);
	}

	@Test
	public void removeEmptyRowsEmptySelect() {
		FrameBlock in = getTestCase2();
		MatrixBlock select = new MatrixBlock(10, 1, false);
		FrameBlock out = in.removeEmptyOperations(true, true, select);
		assertEquals(2, out.getNumColumns());
		assertEquals(1, out.getNumRows());
		assertEquals(null, out.get(0, 0));
	}

	@Test
	public void removeEmptyRowsSelect() {
		FrameBlock in = getTestCase2();
		MatrixBlock select = new MatrixBlock(10, 1, false);
		select.setValue(3, 0, 1);
		FrameBlock out = in.removeEmptyOperations(true, true, select);
		assertEquals(2, out.getNumColumns());
		assertEquals(1, out.getNumRows());
		assertEquals("Hi", out.get(0, 0));
	}

	@Test
	public void removeEmptyRowsSelect_2() {
		FrameBlock in = getTestCase2();
		MatrixBlock select = new MatrixBlock(10, 1, false);
		select.setValue(3, 0, 1);
		select.setValue(4, 0, 1);
		FrameBlock out = in.removeEmptyOperations(true, true, select);
		assertEquals(2, out.getNumColumns());
		assertEquals(2, out.getNumRows());
		assertEquals("Hi", out.get(0, 0));
		assertEquals("You", out.get(1, 1));
	}

	@Test
	public void removeEmptyRows() {
		FrameBlock in = getTestCase2();
		FrameBlock out = in.removeEmptyOperations(true, true, null);
		assertEquals(2, out.getNumColumns());
		assertEquals(2, out.getNumRows());
		assertEquals("Hi", out.get(0, 0));
		assertEquals("You", out.get(1, 1));
	}

	@Test
	public void removeEmptyRowsEmptyIn() {
		FrameBlock in = getTestCase0();
		FrameBlock out = in.removeEmptyOperations(true, true, null);
		assertEquals(2, out.getNumColumns());
		assertEquals(1, out.getNumRows());
	}

	@Test
	public void removeEmptyRowsEmptyInNotEmptyReturn() {
		FrameBlock in = getTestCase0();
		FrameBlock out = in.removeEmptyOperations(true, false, null);
		assertEquals(2, out.getNumColumns());
		assertEquals(0, out.getNumRows());
	}

	@Test
	public void removeEmptyRowsEmptyInColumnName() {
		try{

			FrameBlock in = getTestCase0();
			in.setColumnName(1, "HelloThere");
			FrameBlock out = in.removeEmptyOperations(true, true, null);
			assertEquals(2, out.getNumColumns());
			assertEquals(1, out.getNumRows());
			assertEquals("HelloThere", out.getColumnName(1));
		}
		catch(Exception e){
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void removeEmptyRowsEmptySelectAll() {
		FrameBlock in = getTestCase0();
		in.setColumnName(1, "HelloThere");
		MatrixBlock select = new MatrixBlock(10, 1, 1.0);
		select.recomputeNonZeros();
		FrameBlock out = in.removeEmptyOperations(true, true, select);
		assertEquals(2, out.getNumColumns());
		assertEquals(10, out.getNumRows());
		assertEquals("HelloThere", out.getColumnName(1));
	}

	@Test
	public void removeEmptyRowsEmptyAll() {
		FrameBlock in = getTestCase3();
		in.setColumnName(1, "HelloThere");
		FrameBlock out = in.removeEmptyOperations(true, true, null);
		assertEquals(2, out.getNumColumns());
		assertEquals(10, out.getNumRows());
		assertEquals("HelloThere", out.getColumnName(1));
	}

	@Test
	public void removeEmptyRowsSomeColumnsWithName() {
		FrameBlock in = getTestCase2();
		in.setColumnName(1, "HelloThere");
		FrameBlock out = in.removeEmptyOperations(true, true, null);
		assertEquals(2, out.getNumColumns());
		assertEquals(2, out.getNumRows());
		assertEquals("HelloThere", out.getColumnName(1));
	}

	@Test
	public void removeEmptyRowsSelectAlmostAll() {
		FrameBlock in = getTestCase2();
		in.setColumnName(1, "HelloThere");
		MatrixBlock select = new MatrixBlock(10, 1, 1.0);
		select.setValue(0, 0, 0);
		FrameBlock out = in.removeEmptyOperations(true, true, select);
		assertEquals(2, out.getNumColumns());
		assertEquals(9, out.getNumRows());
		assertEquals("HelloThere", out.getColumnName(1));
	}

	@Test
	public void removeEmptyColsWithSelectNone() {
		try {
			FrameBlock in = getTestCase2();
			MatrixBlock select = new MatrixBlock(1, 2, new double[] {0, 0});
			select.recomputeNonZeros();
			FrameBlock out = in.removeEmptyOperations(false, false, select);
			assertEquals(0, out.getNumColumns());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void removeEmptyColsWithSelectNoneEmptyReturn() {
		try {
			FrameBlock in = getTestCase2();
			MatrixBlock select = new MatrixBlock(1, 2, new double[] {0, 0});
			select.recomputeNonZeros();
			FrameBlock out = in.removeEmptyOperations(false, true, select);
			assertEquals(1, out.getNumColumns());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private static FrameBlock getTestCase0() {
		FrameBlock in = new FrameBlock();
		in.appendColumn(ArrayFactory.allocate(ValueType.STRING, 10));
		in.appendColumn(ArrayFactory.allocate(ValueType.STRING, 10));
		return in;
	}

	private static FrameBlock getTestCase1() {
		FrameBlock in = new FrameBlock();
		in.appendColumn(ArrayFactory.allocate(ValueType.STRING, 10));
		in.set(3, 0, "Hi");
		return in;
	}

	private static FrameBlock getTestCase2() {
		FrameBlock in = new FrameBlock();
		in.appendColumn(ArrayFactory.allocate(ValueType.STRING, 10));
		in.set(3, 0, "Hi");
		in.appendColumn(ArrayFactory.allocate(ValueType.STRING, 10));
		in.set(4, 1, "You");
		return in;
	}

	private static FrameBlock getTestCase3() {
		FrameBlock in = new FrameBlock();
		in.appendColumn(ArrayFactory.allocate(ValueType.STRING, 10, "Hi"));
		in.appendColumn(ArrayFactory.allocate(ValueType.STRING, 10, "You"));
		return in;
	}
}
