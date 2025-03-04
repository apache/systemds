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
import org.apache.sysds.runtime.compress.lib.CLALibRexpand;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class SeqTableTest {

	protected static final Log LOG = LogFactory.getLog(SeqTableTest.class.getName());

	static{
		CLALibRexpand.ALLOW_COMPRESSED_TABLE_SEQ = false; // allow the compressed tables.
	}

	@Test(expected = Exception.class)
	public void test_notSameDim() {
		MatrixBlock c = new MatrixBlock(20, 1, 0.0);
		LibMatrixReorg.fusedSeqRexpand(10, c, 1);
	}

	@Test(expected = Exception.class)
	public void test_toLow() {
		MatrixBlock c = new MatrixBlock(10, 1, -1.0);
		LibMatrixReorg.fusedSeqRexpand(10, c, 1);
	}

	@Test(expected = Exception.class)
	public void test_toManyColumn() {
		MatrixBlock c = new MatrixBlock(10, 2, -1.0);
		LibMatrixReorg.fusedSeqRexpand(10, c, 1);
	}

	@Test
	public void test_All_NaN() {
		MatrixBlock c = new MatrixBlock(10, 1, Double.NaN);
		MatrixBlock ret = LibMatrixReorg.fusedSeqRexpand(10, c, 1);

		assertEquals(0, ret.getNumColumns());
	}

	@Test
	public void test_w_NaN() {
		MatrixBlock c = new MatrixBlock(10, 1, 1.0);
		MatrixBlock ret = LibMatrixReorg.fusedSeqRexpand(10, c, Double.NaN);
		assertEquals(0, ret.getNumColumns());
	}

	@Test
	public void test_all_one() {
		MatrixBlock c = new MatrixBlock(10, 1, 1.0);
		MatrixBlock ret = LibMatrixReorg.fusedSeqRexpand(10, c, 1);
		assertEquals(1, ret.getNumColumns());
		TestUtils.compareMatrices(c, ret, 0);
	}

	@Test
	public void test_all_one_givenMatrixBlock() {
		MatrixBlock c = new MatrixBlock(10, 1, 1.0);
		MatrixBlock ret = LibMatrixReorg.fusedSeqRexpand(10, c, 1, new MatrixBlock(), true);
		assertEquals(1, ret.getNumColumns());
		TestUtils.compareMatrices(c, ret, 0);
	}

	@Test
	public void test_all_one_givenMatrixBlockWithSize() {
		MatrixBlock c = new MatrixBlock(10, 1, 1.0);
		MatrixBlock ret = LibMatrixReorg.fusedSeqRexpand(10, c, 1, new MatrixBlock(1,2, 0.0), false);
		assertEquals(2, ret.getNumColumns());
		MatrixBlock expected = c.append(new MatrixBlock(10, 1, 0.0));
		TestUtils.compareMatrices(expected, ret, 0);
	}

	@Test
	public void test_all_one_givenMatrixBlockWithSize_NaNWeight() {
		MatrixBlock c = new MatrixBlock(10, 1, 1.0);
		MatrixBlock ret = LibMatrixReorg.fusedSeqRexpand(10, c, Double.NaN, new MatrixBlock(1,2, 0.0), false);
		assertEquals(2, ret.getNumColumns());
		MatrixBlock expected = new MatrixBlock(10, 2, 0.0);
		TestUtils.compareMatrices(expected, ret, 0);
	}
}
