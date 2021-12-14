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

package org.apache.sysds.test.component.compress.estim.encoding;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public abstract class EncodeSampleTest {

	protected static final Log LOG = LogFactory.getLog(EncodeSampleTest.class.getName());

	public MatrixBlock m;
	public boolean t;
	public int u;
	public IEncode e;

	public EncodeSampleTest(MatrixBlock m, boolean t, int u, IEncode e) {
		this.m = m;
		this.t = t;
		this.u = u;
		this.e = e;
	}

	@Test
	public void getUnique() {
		if(u != e.getUnique()) {
			StringBuilder sb = new StringBuilder();
			sb.append("invalid number of unique expected:");
			sb.append(u);
			sb.append(" got: ");
			sb.append(e.getUnique());
			sb.append("\n");
			sb.append(e);
			fail(sb.toString());
		}
	}

	@Test
	public void testToString() {
		e.toString();
	}

	@Test
	public void testJoinSelfEqualsSameNumberUnique() {
		try {
			// not that you should or would ever do this.
			// but it is a nice and simple test.
			IEncode j = e.join(e);
			if(u != j.getUnique()) {
				StringBuilder sb = new StringBuilder();
				sb.append("invalid number of unique expected:");
				sb.append(u);
				sb.append(" got: ");
				sb.append(j.getUnique());
				sb.append("\nexpected encoding:\n");
				sb.append(e);
				sb.append("\ngot\n:");
				sb.append(j);
				fail(sb.toString());
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testJoinEmptyLeft() {
		try {
			final MatrixBlock empty = new MatrixBlock(m.getNumRows(), m.getNumColumns(), true);
			final IEncode emptyEncoding = IEncode.createFromMatrixBlock(empty, t, 0);
			assertEquals(u, emptyEncoding.join(e).getUnique());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testJoinEmptyRight() {
		try {
			final MatrixBlock empty = new MatrixBlock(m.getNumRows(), m.getNumColumns(), true);
			final IEncode emptyEncoding = IEncode.createFromMatrixBlock(empty, t, 0);
			assertEquals(u, e.join(emptyEncoding).getUnique());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testJoinConstLeft() {
		try {
			final MatrixBlock c = new MatrixBlock(m.getNumRows(), m.getNumColumns(), 1.0);
			final IEncode emptyEncoding = IEncode.createFromMatrixBlock(c, t, 0);
			assertEquals(u, emptyEncoding.join(e).getUnique());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testJoinConstRight() {
		try {
			final MatrixBlock c = new MatrixBlock(m.getNumRows(), m.getNumColumns(), 1.0);

			final IEncode emptyEncoding = IEncode.createFromMatrixBlock(c, t, 0);
			assertEquals(u, e.join(emptyEncoding).getUnique());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testGetSize() {
		try {
			assertTrue(e.size() <= (t ? m.getNumColumns() : m.getNumRows()));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testGetSizeAfterJoinSelf() {
		try {
			assertTrue(e.join(e).size() <= (t ? m.getNumColumns() : m.getNumRows()));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void toEstimationFactors() {
		try {
			int[] cols = new int[t ? m.getNumRows() : m.getNumColumns()];
			int rows = t ? m.getNumColumns() : m.getNumRows();
			e.computeSizeEstimation(cols, rows, 1.0, 1.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
}
