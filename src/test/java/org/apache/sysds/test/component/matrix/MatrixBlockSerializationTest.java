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

import static org.junit.Assert.fail;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class MatrixBlockSerializationTest {

	private MatrixBlock mb;

	@Parameters
	public static Collection<Object[]> data() {
		List<Object[]> tests = new ArrayList<>();

		try {
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 1.0, 3)});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1000, 100, 0, 10, 1.0, 3)});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 1, 0, 10, 1.0, 3)});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1, 100, 0, 10, 1.0, 3)});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 10, 0, 10, 1.0, 3)});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 1000, 0, 10, 1.0, 3)});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1000, 1000, 0, 10, 1.0, 3)});

			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.1, 3)});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1000, 100, 0, 10, 0.1, 3)});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 1, 0, 10, 0.1, 3)});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1, 100, 0, 10, 0.1, 3)});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 10, 0, 10, 0.1, 3)});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 1000, 0, 10, 0.1, 3)});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1000, 1000, 0, 10, 0.1, 3)});

			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 100, 0, 10, 0.001, 3)});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1000, 100, 0, 10, 0.001, 3)});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 1, 0, 10, 0.001, 3)});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1, 100, 0, 10, 0.001, 3)});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 10, 0, 10, 0.001, 3)});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(100, 1000, 0, 10, 0.001, 3)});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1000, 1000, 0, 10, 0.001, 3)});
			tests.add(new Object[] {new MatrixBlock()});

		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	public MatrixBlockSerializationTest(MatrixBlock mb) {
		this.mb = mb;
	}

	@Test
	public void testSerialization() {
		try {
			// serialize compressed matrix block
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			mb.write(fos);

			// deserialize compressed matrix block
			ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
			DataInputStream fis = new DataInputStream(bis);
			MatrixBlock in = new MatrixBlock();
			in.readFields(fis);
			TestUtils.compareMatrices(mb, in, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
}
