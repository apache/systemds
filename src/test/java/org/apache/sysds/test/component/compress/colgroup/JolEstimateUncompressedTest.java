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

package org.apache.sysds.test.component.compress.colgroup;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/**
 * Test for the Uncompressed Col Group, To verify that we estimate the memory usage to be the worst case possible.
 */
@RunWith(value = Parameterized.class)
public class JolEstimateUncompressedTest extends JolEstimateTest {

	@Parameters
	public static Collection<Object[]> data() {

		ArrayList<Object[]> tests = new ArrayList<>();
		ArrayList<MatrixBlock> mb = new ArrayList<>();

		mb.add(DataConverter.convertToMatrixBlock(new double[][] {{0}}));
		mb.add(DataConverter.convertToMatrixBlock(new double[][] {{1}}));
		mb.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(1, 100, 0, 100, 1.0, 7)));
		mb.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(1, 1000, 0, 100, 0.2, 7)));
		mb.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(1, 100000, 0, 100, 0.01, 7)));

		// Multi column
		// TODO Fix uncompressed columns in lossy situation
		// mb.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(2, 10, 0, 100, 1.0, 7)));
		// mb.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(13, 100, 0, 100, 1.0, 7)));

		// sparse

		// mb.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(13, 100, 0, 100, 0.3, 7)));
		// mb.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(100, 100, 0, 100, 0.01, 7)));

		for(MatrixBlock m : mb) {
			tests.add(new Object[] {m});
		}

		return tests;
	}

	public JolEstimateUncompressedTest(MatrixBlock mb) {
		super(mb, 0);
	}

	@Override
	public CompressionType getCT() {
		return unc;
	}

}