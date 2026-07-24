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

import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
public class JolEstimateDeltaDDCTest extends JolEstimateTest {

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		MatrixBlock mb;

		mb = DataConverter.convertToMatrixBlock(new double[][] {{0}});
		tests.add(new Object[] {mb});

		mb = DataConverter.convertToMatrixBlock(new double[][] {{1}});
		tests.add(new Object[] {mb});

		mb = DataConverter.convertToMatrixBlock(new double[][] {{1, 2, 3, 4, 5}});
		tests.add(new Object[] {mb});

		mb = DataConverter.convertToMatrixBlock(new double[][] {{1,2,3},{1,1,1}});
		tests.add(new Object[] {mb});

		mb = DataConverter.convertToMatrixBlock(new double[][] {{1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}});
		tests.add(new Object[] {mb});

		mb = TestUtils.generateTestMatrixBlock(2, 5, 0, 20, 1.0, 7);
		tests.add(new Object[] {mb});

		return tests;
	}

	public JolEstimateDeltaDDCTest(MatrixBlock mb) {
		super(mb);
	}

	@Override
	public AColGroup.CompressionType getCT() {
		return delta;
	}

	@Override
	protected boolean shouldTranspose() {
		return false;
	}
}
