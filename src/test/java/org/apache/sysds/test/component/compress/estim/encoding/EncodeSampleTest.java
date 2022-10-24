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
import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
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
	public void testCombineSelfEqualsSameNumberUnique() {
		try {
			// not that you should or would ever do this.
			// but it is a nice and simple test.
			IEncode j = e.combine(e);
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
	public void testCombineEmptyLeft() {
		try {
			final MatrixBlock empty = new MatrixBlock(m.getNumRows(), m.getNumColumns(), true);
			final IEncode emptyEncoding = EncodingFactory.createFromMatrixBlock(empty, t, 0);
			assertEquals(u, emptyEncoding.combine(e).getUnique());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testCombineEmptyRight() {
		try {
			final MatrixBlock empty = new MatrixBlock(m.getNumRows(), m.getNumColumns(), true);
			final IEncode emptyEncoding = EncodingFactory.createFromMatrixBlock(empty, t, 0);
			assertEquals(u, e.combine(emptyEncoding).getUnique());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testCombineConstLeft() {
		try {
			final MatrixBlock c = new MatrixBlock(m.getNumRows(), m.getNumColumns(), 1.0);
			final IEncode emptyEncoding = EncodingFactory.createFromMatrixBlock(c, t, 0);
			assertEquals(u, emptyEncoding.combine(e).getUnique());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testCombineConstRight() {
		try {
			final MatrixBlock c = new MatrixBlock(m.getNumRows(), m.getNumColumns(), 1.0);
			final IEncode emptyEncoding = EncodingFactory.createFromMatrixBlock(c, t, 0);
			final IEncode comp = e.combine(emptyEncoding);
			assertEquals(u, comp.getUnique());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void toEstimationFactors() {
		try {
			int rows = t ? m.getNumColumns() : m.getNumRows();
			EstimationFactors a = e.extractFacts(rows, 1.0, 1.0, new CompressionSettingsBuilder().create());
			int[] f = a.getFrequencies();
			if(f != null)
				for(int i : f)
					if(i <= 0)
						fail("Frequencies contains zero");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void toEstimationFactorsWithRLE() {
		try {
			int rows = t ? m.getNumColumns() : m.getNumRows();
			EstimationFactors a = e.extractFacts(rows, 1.0, 1.0, new CompressionSettingsBuilder().addValidCompression(CompressionType.RLE).create());
			int[] f = a.getFrequencies();
			if(f != null)
				for(int i : f)
					if(i <= 0)
						fail("Frequencies contains zero");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void isDense(){
		boolean d = e.isDense();
		int rows = t ? m.getNumColumns() : m.getNumRows();
		if(rows == 1 && m.isInSparseFormat() && ! d)
			fail ("Should extract sparse if input is sparse and one column (row)");
	}
}
