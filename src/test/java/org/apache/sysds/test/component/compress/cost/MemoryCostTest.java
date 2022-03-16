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

package org.apache.sysds.test.component.compress.cost;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;

import org.apache.sysds.runtime.compress.cost.MemoryCostEstimator;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class MemoryCostTest extends ACostTest {

	public MemoryCostTest(MatrixBlock mb, int seed) {
		super(mb, new MemoryCostEstimator(), seed);
	}


	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		Random r = new Random(1231);
		final int m = Integer.MAX_VALUE;
		tests.add(new Object[]{gen(1000, 2, 0, 5, 0.1, r), r.nextInt(m)});
		tests.add(new Object[]{gen(3000, 10, 0, 5, 0.1, r), r.nextInt(m)});
		tests.add(new Object[]{gen(3000, 10, 0, 5, 1.0, r), r.nextInt(m)});
		return tests;
	}


	private static MatrixBlock gen(int nRow, int nCol, int min, int max, double s,  Random r){
		final int m = Integer.MAX_VALUE;
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(nRow, nCol, min, max, s, r.nextInt(m));
		return TestUtils.round(mb);
	}

}
