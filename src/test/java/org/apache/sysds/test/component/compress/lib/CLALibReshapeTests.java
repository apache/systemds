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

package org.apache.sysds.test.component.compress.lib;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.lib.CLALibReshape;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class CLALibReshapeTests {
	protected static final Log LOG = LogFactory.getLog(CLALibReshapeTests.class.getName());

	static{
		Thread.currentThread().setName("test_reshape");
	}

	@Test
	public void reshapeSimple() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(1000, 5, 1, 1, 0.5, 235);
		MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb).getLeft();

		MatrixBlock m3 = CLALibReshape.reshape((CompressedMatrixBlock) m2, 500, 10, false);
		MatrixBlock ref = LibMatrixReorg.reshape(mb, 500, 10, false);

		TestUtils.compareMatrices(ref, m3, 0);
	}

	@Test
	public void reshapeSimple2Rowwise() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(3000, 1, 1, 1, 0.5, 235);
		MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb).getLeft();

		MatrixBlock m3 = CLALibReshape.reshape((CompressedMatrixBlock) m2, 1500, 2, true);
		MatrixBlock ref = LibMatrixReorg.reshape(mb, 1500, 2, true);

		TestUtils.compareMatrices(ref, m3, 0);
	}

	@Test
	public void reshapeMulti2Rowwise() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(3000, 4, 1, 1, 0.5, 235);
		MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb).getLeft();

		MatrixBlock m3 = CLALibReshape.reshape((CompressedMatrixBlock) m2, 1500, 8, true);
		MatrixBlock ref = LibMatrixReorg.reshape(mb, 1500, 8, true);

		TestUtils.compareMatrices(ref, m3, 0);
	}


	@Test
	public void reshapeMulti2RowwiseSingleThread() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(3000, 4, 1, 1, 0.5, 235);
		MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb).getLeft();

		MatrixBlock m3 = CLALibReshape.reshape((CompressedMatrixBlock) m2, 1500, 8, true, 1);
		MatrixBlock ref = LibMatrixReorg.reshape(mb, 1500, 8, true);

		TestUtils.compareMatrices(ref, m3, 0);
	}

	@Test
	public void reshapeSimple2RowwiseNotMultiply() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(3000, 2, 1, 1, 0.5, 235);
		MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb).getLeft();

		MatrixBlock m3 = CLALibReshape.reshape((CompressedMatrixBlock) m2, 2000, 3, true);
		MatrixBlock ref = LibMatrixReorg.reshape(mb, 2000, 3, true);

		TestUtils.compareMatrices(ref, m3, 0);
	}

	@Test
	public void reshapeSimple2ColWise() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(3000, 1, 1, 1, 0.5, 235);
		MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb).getLeft();

		MatrixBlock m3 = CLALibReshape.reshape((CompressedMatrixBlock) m2, 1500, 2, false);
		MatrixBlock ref = LibMatrixReorg.reshape(mb, 1500, 2, false);

		TestUtils.compareMatrices(ref, m3, 0);
	}

	@Test(expected = Exception.class)
	public void reshapeInvalid() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(1000, 5, 1, 1, 0.5, 235);
		MatrixBlock m2 = CompressedMatrixBlockFactory.compress(mb).getLeft();

		CLALibReshape.reshape((CompressedMatrixBlock) m2, 501, 10, false);
	}
}
