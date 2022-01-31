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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.api.DMLException;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.LibMatrixCountDistinct;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperator;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperatorTypes;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Hash.HashType;
import org.junit.Assert;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import java.util.ArrayList;
import java.util.Collection;

@RunWith(value = Parameterized.class)
public class CountDistinctTest {

	private static CountDistinctOperatorTypes[] esT = new CountDistinctOperatorTypes[] {
		// The different types of Estimators
		CountDistinctOperatorTypes.COUNT, CountDistinctOperatorTypes.KMV, CountDistinctOperatorTypes.HLL};

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		ArrayList<MatrixBlock> inputs = new ArrayList<>();
		ArrayList<Long> actualUnique = new ArrayList<>();

		// single value matrix.
		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(1, 1, 0.0, 100.0, 1, 7)));
		actualUnique.add(1L);

		// single column or row matrix.
		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(1, 100, 0.0, 100.0, 1, 7)));
		actualUnique.add(100L);
		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(100, 1, 0.0, 100.0, 1, 7)));
		actualUnique.add(100L);

		// Sparse Multicol random values (most likely each value is unique)
		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(100, 10, 0.0, 100.0, 0.1, 7)));
		actualUnique.add(98L); // dense representation
		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(100, 1000, 0.0, 100.0, 0.1, 7)));
		actualUnique.add(9823L + 1); // sparse representation

		// MultiCol Inputs (using integers)
		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrixIntV(5000, 5000, 1, 100, 1, 8)));
		actualUnique.add(99L);
		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrixIntV(1024, 10240, 1, 100, 1, 7)));
		actualUnique.add(99L);
		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrixIntV(10240, 1024, 1, 100, 1, 7)));
		actualUnique.add(99L);
		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrixIntV(1024, 10241, 1, 1500, 1, 7)));
		actualUnique.add(1499L);
		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrixIntV(1024, 10241, 0, 3000, 1, 7)));
		actualUnique.add(3000L);
		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrixIntV(1024, 10241, 0, 6000, 1, 7)));
		actualUnique.add(6000L);

		// Sparse Inputs
		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrixIntV(1024, 10241, 0, 3000, 0.1, 7)));
		actualUnique.add(3000L);

		for(CountDistinctOperatorTypes et : esT) {
			for(HashType ht : HashType.values()) {
				if((ht == HashType.ExpHash && et == CountDistinctOperatorTypes.KMV) ||
					(ht == HashType.StandardJava && et == CountDistinctOperatorTypes.KMV)) {
					String errorMessage = "Invalid hashing configuration using " + ht + " and " + et;
					tests.add(new Object[] {et, inputs.get(0), actualUnique.get(0), ht, DMLException.class,
						errorMessage, 0.0});
				}
				else if(et == CountDistinctOperatorTypes.HLL) {
					tests.add(new Object[] {et, inputs.get(0), actualUnique.get(0), ht, NotImplementedException.class,
						"HyperLogLog not implemented", 0.0});
				}
				else if(et != CountDistinctOperatorTypes.COUNT) {
					for(int i = 0; i < inputs.size(); i++) {
						// allowing the estimate to be 15% off
						tests.add(new Object[] {et, inputs.get(i), actualUnique.get(i), ht, null, null, 0.15});
					}
				}
			}
			if(et == CountDistinctOperatorTypes.COUNT) {
				for(int i = 0; i < inputs.size(); i++) {
					tests.add(new Object[] {et, inputs.get(i), actualUnique.get(i), null, null, null, 0.0001});
				}
			}
		}
		return tests;
	}

	@Parameterized.Parameter
	public CountDistinctOperatorTypes et;
	@Parameterized.Parameter(1)
	public MatrixBlock in;
	@Parameterized.Parameter(2)
	public long nrUnique;
	@Parameterized.Parameter(3)
	public HashType ht;

	// Exception handling
	@Parameterized.Parameter(4)
	public Class<? extends Exception> expectedException;
	@Parameterized.Parameter(5)
	public String expectedExceptionMsg;

	@Rule
	public ExpectedException thrown = ExpectedException.none();

	// allowing the estimate to be within 20% of target.
	@Parameterized.Parameter(6)
	public double epsilon;

	@Test
	public void testEstimation() {

		// setup expected exception
		if(expectedException != null) {
			thrown.expect(expectedException);
			thrown.expectMessage(expectedExceptionMsg);
		}

		Integer out = 0;
		CountDistinctOperator op = new CountDistinctOperator(et, ht)
				.setDirection(Types.Direction.RowCol);
		try {
			out = LibMatrixCountDistinct.estimateDistinctValues(in, op);
		}
		catch(DMLException e) {
			throw e;
		}
		catch(NotImplementedException e) {
			throw e;
		}
		catch(Exception e) {
			e.printStackTrace();
			Assert.assertTrue(this.toString(), false);
		}

		int count = out;
		boolean success = Math.abs(nrUnique - count) <= nrUnique * epsilon;
		StringBuilder sb = new StringBuilder();
		sb.append(this.toString());
		sb.append("\n" + count + " unique values, actual:" + nrUnique + " with eps of " + epsilon);
		Assert.assertTrue(sb.toString(), success);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(et);
		if(ht != null) {
			sb.append("-" + ht);
		}
		sb.append("  nrUnique:" + nrUnique);
		sb.append(" & input size:" + in.getNumRows() + "," + in.getNumColumns());
		sb.append(" sparse: " + in.isInSparseFormat());
		if(expectedException != null) {
			sb.append("\nExpected Exception: " + expectedException);
			sb.append("\nExpected Exception Msg: " + expectedExceptionMsg);
		}
		return sb.toString();
	}
}
