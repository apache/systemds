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

package org.apache.sysds.test.component.compress;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.functionobjects.CM;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator.AggregateOperationTypes;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.TestConstants.MatrixTypology;
import org.apache.sysds.test.component.compress.TestConstants.SparsityType;
import org.apache.sysds.test.component.compress.TestConstants.ValueRange;
import org.apache.sysds.test.component.compress.TestConstants.ValueType;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class CompressedVectorTest extends CompressedTestBase {

	protected static MatrixTypology[] usedMatrixTypologyLocal = new MatrixTypology[] {// types
		MatrixTypology.SINGLE_COL,
		// MatrixTypology.SINGLE_COL_L
	};

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		for(SparsityType st : usedSparsityTypes) {
			for(ValueType vt : usedValueTypes) {
				for(ValueRange vr : usedValueRanges) {
					for(CompressionSettings cs : usedCompressionSettings) {
						for(MatrixTypology mt : usedMatrixTypologyLocal) {
							tests.add(new Object[] {st, vt, vr, cs, mt});
						}
					}
				}
			}
		}
		return tests;
	}

	public CompressedVectorTest(SparsityType sparType, ValueType valType, ValueRange valRange,
		CompressionSettings compSettings, MatrixTypology matrixTypology) {
		super(sparType, valType, valRange, compSettings, matrixTypology);
	}

	@Test
	public void testCentralMoment() throws Exception {
		// TODO: Make Central Moment Test work on Multi dimensional Matrix
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			// quantile uncompressed
			AggregateOperationTypes opType = CMOperator.getCMAggOpType(2);
			CMOperator cm = new CMOperator(CM.getCMFnObject(opType), opType);

			double ret1 = mb.cmOperations(cm).getRequiredResult(opType);

			// quantile compressed
			double ret2 = cmb.cmOperations(cm).getRequiredResult(opType);
			// compare result with input allowing 1 bit difference in least significant location
			TestUtils.compareScalarBitsJUnit(ret1, ret2, 64);

		}
		catch(Exception e) {
			throw new Exception(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testQuantile() {
		try {
			// quantile uncompressed
			MatrixBlock tmp1 = mb.sortOperations(null, new MatrixBlock());
			double ret1 = tmp1.pickValue(0.95);

			// quantile compressed
			MatrixBlock tmp2 = cmb.sortOperations(null, new MatrixBlock());
			double ret2 = tmp2.pickValue(0.95);

			// compare result with input
			TestUtils.compareScalarBitsJUnit(ret1, ret2, 64);
		}
		catch(Exception e) {
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}
}
