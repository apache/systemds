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

package org.apache.sysds.test.component.matrix;

import static org.junit.Assert.assertEquals;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.junit.Test;

public class SparseCSRTest {
	protected static final Log LOG = LogFactory.getLog(CompressedMatrixBlock.class.getName());

	@Test
	public void testGTE() {
		int[] rs = new int[] {0, 9};
		int[] colInd = new int[] {10, 20, 30, 40, 50, 60, 80, 90, 100};
		double[] val = new double[] {1, 1, 1, 1, 1, 1, 1, 1, 1};
		SparseBlockCSR b = new SparseBlockCSR(rs, colInd, val, val.length);

		assertEquals(0, b.posFIndexGTE(0, 0));
		assertEquals(0, b.posFIndexGTE(0, 10));
		assertEquals(1, b.posFIndexGTE(0, 11));
		assertEquals(7, b.posFIndexGTE(0, 90));
		assertEquals(8, b.posFIndexGTE(0, 91));
		assertEquals(-1, b.posFIndexGTE(0, 101));
		assertEquals(-1, b.posFIndexGTE(0, 10100));

	}

	@Test
	public void testGTE2Rows() {
		int[] rs = new int[] {0, 0, 9};
		int[] colInd = new int[] {10, 20, 30, 40, 50, 60, 80, 90, 100};
		double[] val = new double[] {1, 1, 1, 1, 1, 1, 1, 1, 1};
		SparseBlockCSR b = new SparseBlockCSR(rs, colInd, val, val.length);
		LOG.error(b);

		assertEquals(0, b.posFIndexGTE(1, 0));
		assertEquals(0, b.posFIndexGTE(1, 10));
		assertEquals(1, b.posFIndexGTE(1, 11));
		assertEquals(7, b.posFIndexGTE(1, 90));
		assertEquals(8, b.posFIndexGTE(1, 91));
		assertEquals(-1, b.posFIndexGTE(1, 101));
		assertEquals(-1, b.posFIndexGTE(1, 10100));

	}

	@Test
	public void testGTE2RowsNN() {
		int[] rs = new int[] {0, 1, 10};
		int[] colInd = new int[] {100, 10, 20, 30, 40, 50, 60, 80, 90, 100};
		double[] val = new double[] {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
		SparseBlockCSR b = new SparseBlockCSR(rs, colInd, val, val.length);
		LOG.error(b);

		assertEquals(0, b.posFIndexGTE(1, 0));
		assertEquals(0, b.posFIndexGTE(1, 10));
		assertEquals(1, b.posFIndexGTE(1, 11));
		assertEquals(7, b.posFIndexGTE(1, 90));
		assertEquals(8, b.posFIndexGTE(1, 91));
		assertEquals(-1, b.posFIndexGTE(1, 101));
		assertEquals(-1, b.posFIndexGTE(1, 10100));

	}
}
