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

package org.apache.sysds.test.component.sparse;

import java.lang.reflect.Field;

import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public class SparseBlockCompactTest extends AutomatedTestBase
{
	private final static int _rows = 324;
	private final static int _cols = 132;
	private final static double _sparsity = 0.22;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testSparseBlockCompactCOO()  {
		runSparseBlockCompactZerosTest(SparseBlock.Type.COO);
	}

	@Test
	public void testSparseBlockCompactCSC()  {
		runSparseBlockCompactZerosTest(SparseBlock.Type.CSC);
	}

	@Test
	public void testSparseBlockCompactCSR()  {
		runSparseBlockCompactZerosTest(SparseBlock.Type.CSR);
	}

	@Test
	public void testSparseBlockCompactDCSR()  {
		runSparseBlockCompactZerosTest(SparseBlock.Type.DCSR);
	}

	@Test
	public void testSparseBlockCompactMCSC()  {
		runSparseBlockModifiedCompactZerosTest(SparseBlock.Type.MCSC, "_columns");
	}

	@Test
	public void testSparseBlockCompactMCSR()  {
		runSparseBlockModifiedCompactZerosTest(SparseBlock.Type.MCSR, "_rows");
	}

	private void runSparseBlockCompactZerosTest(SparseBlock.Type btype) {

		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 13);

		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp = mbtmp.getSparseBlock();
		SparseBlock sblock = SparseBlockFactory.copySparseBlock(btype, srtmp, true);

		double[] values = (double[]) getField(sblock, "_values");
		values[0] = 0.0;
		values[values.length-1] = 0.0;

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(_rows, _cols, sblock.size(), true));
		assertTrue(ex.getMessage().startsWith("The values array should not contain zeros"));
		long size = sblock.size();

		sblock.compact();

		assertTrue("should pass checkValidity", sblock.checkValidity(_rows, _cols, sblock.size(), true));
		assertEquals(size-2, sblock.size());
	}

	private void runSparseBlockModifiedCompactZerosTest(SparseBlock.Type btype, String field) {

		double[][] A = getRandomMatrix(_rows, _cols, -10, 10, _sparsity, 13);

		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
		SparseBlock srtmp = mbtmp.getSparseBlock();
		SparseBlock sblock = SparseBlockFactory.copySparseBlock(btype, srtmp, true);

		SparseRow[] sr = (SparseRow[]) getField(sblock, field);
		double[] values = sr[0].values();
		values[0] = 0.0;
		values[values.length-1] = 0.0;

		RuntimeException ex = assertThrows(RuntimeException.class,
			() -> sblock.checkValidity(_rows, _cols, sblock.size(), true));
		assertTrue(ex.getMessage().startsWith("The values array should not contain zeros"));
		long size = sblock.size();

		sblock.compact();

		assertTrue("should pass checkValidity", sblock.checkValidity(_rows, _cols, sblock.size(), true));
		assertEquals(size-2, sblock.size());
	}

	private static Object getField(Object obj, String name) {
		try {
			Field f = obj.getClass().getDeclaredField(name);
			f.setAccessible(true);
			return f.get(obj);
		} catch (Exception ex) {
			throw new RuntimeException("Reflection failed: " + ex.getMessage());
		}
	}
}
