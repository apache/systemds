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

@RunWith(value = Parameterized.class)
public class JolEstimateDDCTest extends JolEstimateTest{

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		MatrixBlock mb;

		// Default behavior is to ignore all zero values.
		// because the other compression techniques just ignores their locations
		// DCC is different in that it is a dense compression
		// that also encode 0 values the same as all the other values.

		mb = DataConverter.convertToMatrixBlock(new double[][] {{0}});
		tests.add(new Object[] {mb, new int[]{1}, 8});

		mb = DataConverter.convertToMatrixBlock(new double[][] {{1}});
		tests.add(new Object[] {mb, new int[]{1}, 0});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{1, 2}});
		tests.add(new Object[] {mb, new int[]{2}, 0});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{1, 2, 3}});
		tests.add(new Object[] {mb, new int[]{3}, 0});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{1, 2, 3, 4}});
		tests.add(new Object[] {mb, new int[]{4}, 0});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{1, 2, 3, 4, 5}});
		tests.add(new Object[] {mb, new int[]{5}, 0});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{1, 2, 3, 4, 5, 6}});
		tests.add(new Object[] {mb, new int[]{6}, 0});

		// Dense Random
		mb = DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(1, 20, 0, 20, 1.0, 7));
		tests.add(new Object[] {mb, new int[]{20}, 0});
		mb = DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(1, 100, 0, 20, 1.0, 7));
		tests.add(new Object[] {mb, new int[]{100}, 0});
		mb = DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(1, 500, 0, 20, 1.0, 7));
		tests.add(new Object[] {mb, new int[]{500}, 0});

		// Random Sparse Very big, because 0 is materialized.
		mb = DataConverter
			.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 4000, 0, 254, 0.01, 7)));
		tests.add(new Object[] {mb, new int[]{45}, 8});
		mb = DataConverter
			.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 8000, 0, 254, 0.01, 7)));
		tests.add(new Object[] {mb, new int[]{73}, 8});
		mb = DataConverter
			.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 16000, 0, 254, 0.01, 7)));
		tests.add(new Object[] {mb, new int[]{120}, 8});

		mb = DataConverter
			.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 4000, 0, 254, 0.001, 7)));
		tests.add(new Object[] {mb, new int[]{6}, 8});
		mb = DataConverter
			.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 8000, 0, 254, 0.001, 7)));
		tests.add(new Object[] {mb, new int[]{7}, 8});
		mb = DataConverter
			.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 16000, 0, 254, 0.001, 7)));
		tests.add(new Object[] {mb, new int[]{17}, 8});

		// DDC2 instances, need more unique values than 255

		mb = DataConverter.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 4000, 0, 512, 0.7, 7)));
		tests.add(new Object[] {mb, new int[]{511}, 8});
		mb = DataConverter
			.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 8000, 0, 1024, 0.7, 7)));
		tests.add(new Object[] {mb, new int[]{1020}, 8});
		mb = DataConverter
			.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 16000, 0, 2048, 0.7, 7)));
		tests.add(new Object[] {mb, new int[]{2039}, 8});
		
		return tests;
	}


	public JolEstimateDDCTest(MatrixBlock mb, int[] sizes, int tolerance) {
		super(mb,sizes,tolerance);
	}

	@Override
	public CompressionType getCT() {
		return ddc;
	}

}