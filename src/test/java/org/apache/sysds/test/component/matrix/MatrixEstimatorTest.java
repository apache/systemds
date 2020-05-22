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

import java.util.ArrayList;
import java.util.Collection;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.api.DMLException;
import org.apache.sysds.runtime.instructions.cp.AggregateUnaryCPInstruction.AUType;
import org.apache.sysds.runtime.matrix.data.LibMatrixEstimator;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.EstimatorOperator;
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

@RunWith(value = Parameterized.class)
public class MatrixEstimatorTest {

	private static AUType[] esT = new AUType[] {
		// The different types of Estimators
		AUType.COUNT_DISTINCT,
		// EstimatorType.NUM_DISTINCT_KMV,
		// EstimatorType.NUM_DISTINCT_HYPER_LOG_LOG
	};

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		ArrayList<MatrixBlock> inputs = new ArrayList<>();
		ArrayList<Long> actualUnique = new ArrayList<>();

		// single value matrix.
		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(1, 1, 0.0, 100.0, 1, 7)));
		actualUnique.add(1L);
		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(1, 100, 0.0, 100.0, 1, 7)));
		actualUnique.add(100L);
		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(100, 1, 0.0, 100.0, 1, 7)));
		actualUnique.add(100L);
		// inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(100, 100, 0.0, 100.0, 1, 7)));
		// actualUnique.add(10000L);
		// inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(1024, 1024, 0.0, 100.0, 1, 7)));
		// actualUnique.add(1024L * 1024L);

		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrixIntV(5000, 5000, 1, 100, 1, 8)));
		actualUnique.add(99L);
		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrixIntV(1024, 10240, 1, 100, 1, 7)));
		actualUnique.add(99L);
		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrixIntV(10240, 1024, 1, 100, 1, 7)));
		actualUnique.add(99L);

		// inputs.add(
		// DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrixIntV(1024, 1024, 1000001, 1000100, 1, 8)));
		// actualUnique.add(99L);
		// inputs.add(
		// DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrixIntV(1024, 10240, 1000001, 1000100, 1, 7)));
		// actualUnique.add(99L);

		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrixIntV(1024, 10241, 1, 1500, 1, 7)));
		actualUnique.add(1499L);

		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrixIntV(1024, 10241, 0, 3000, 1, 7)));
		actualUnique.add(3000L);
		inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrixIntV(1024, 10241, 0, 6000, 1, 7)));
		actualUnique.add(6000L);
		// inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrixIntV(10240, 10241, 0, 10000, 1,
		// 7)));
		// actualUnique.add(10000L);
		// inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrixIntV(10240, 10241, 0, 100000, 1,
		// 7)));
		// actualUnique.add(100000L);
		// inputs.add(DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrixIntV(10240, 10241, 0, 1000000, 1,
		// 7)));
		// actualUnique.add(1000000L);

		for(AUType et : esT) {
			for(HashType ht : HashType.values()) {
				if(ht == HashType.ExpHash && et == AUType.COUNT_DISTINCT_ESTIMATE_KMV) {

					String errorMessage = "Invalid hashing configuration using " + HashType.ExpHash + " and "
						+ AUType.COUNT_DISTINCT_ESTIMATE_KMV;
					tests.add(new Object[] {et, inputs.get(0), actualUnique.get(0), ht, DMLException.class,
						errorMessage, 0.0});
				}
				else if(et == AUType.COUNT_DISTINCT_ESTIMATE_HYPER_LOG_LOG) {
					tests.add(new Object[] {et, inputs.get(0), actualUnique.get(0), ht, NotImplementedException.class,
						"HyperLogLog not implemented", 0.0});
				}
				else {
					if(et == AUType.COUNT_DISTINCT) {

						for(int i = 0; i < inputs.size(); i++) {
							tests.add(new Object[] {et, inputs.get(i), actualUnique.get(i), ht, null, null, 0.0001});
						}
					}
					else {
						for(int i = 0; i < inputs.size(); i++) {
							// allowing the estimate to be 30% off
							tests.add(new Object[] {et, inputs.get(i), actualUnique.get(i), ht, null, null, 0.3});
						}
					}
				}
			}

		}

		return tests;
	}

	@Parameterized.Parameter
	public AUType et;
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
		EstimatorOperator op = new EstimatorOperator(et, ht);
		try {
			out = LibMatrixEstimator.estimateDistinctValues(in, op);
		}
		catch(DMLException e) {
			System.out.println(e.getMessage());
			throw e;
		}
		catch(NotImplementedException e) {
			throw e;
		}
		catch(Exception e) {
			e.printStackTrace();
			Assert.assertTrue(
				"EXCEPTION: " + e.getMessage() + " PARAMETERS: " + et + " , hashing: " + ht + " & input size:"
					+ in.getNumRows() + "," + in.getNumColumns(),
				false);
		}

		int count = out;
		boolean success = Math.abs(nrUnique - count) <= nrUnique * epsilon;
		Assert.assertTrue(et + " estimated " + count + " unique values, actual:" + nrUnique + " with eps of " + epsilon
			+ " , hashing: " + ht + " & input size:" + in.getNumRows() + "," + in.getNumColumns(), success);

	}
}