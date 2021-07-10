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

import java.util.Collection;

import org.apache.commons.math3.random.Well1024a;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.LibMatrixDatagen;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.RandomMatrixGenerator;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.TestConstants.MatrixTypology;
import org.apache.sysds.test.component.compress.TestConstants.OverLapping;
import org.apache.sysds.test.component.compress.TestConstants.SparsityType;
import org.apache.sysds.test.component.compress.TestConstants.ValueRange;
import org.apache.sysds.test.component.compress.TestConstants.ValueType;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
public class ParCompressedMatrixTest extends AbstractCompressedUnaryTests {

	public ParCompressedMatrixTest(SparsityType sparType, ValueType valType, ValueRange valRange,
		CompressionSettingsBuilder compressionSettings, MatrixTypology matrixTypology, OverLapping ov,
		Collection<CompressionType> ct) {
		super(sparType, valType, valRange, compressionSettings, matrixTypology, ov, 2, ct);
		// super(sparType, valType, valRange, compressionSettings, matrixTypology, ov,
		// InfrastructureAnalyzer.getLocalParallelism());
	}

	@Override
	public void testUnaryOperators(AggType aggType, boolean inCP) {
		AggregateUnaryOperator auop = super.getUnaryOperator(aggType, _k);
		testUnaryOperators(aggType, auop, inCP);
	}

	@Test
	public void testLeftMatrixMatrixMultMediumSparse2() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			MatrixBlock matrix = DataConverter
				.convertToMatrixBlock(TestUtils.generateTestMatrix(132, rows, 0.9, 1.5, .1, 3));

			// Make Operator
			AggregateBinaryOperator abop = InstructionUtils.getMatMultOperator(_k);

			// vector-matrix uncompressed
			MatrixBlock ret1 = mb.aggregateBinaryOperations(matrix, mb, new MatrixBlock(), abop);

			// vector-matrix compressed
			MatrixBlock ret2 = cmb.aggregateBinaryOperations(matrix, cmb, new MatrixBlock(), abop);

			compareResultMatrices(ret1, ret2, 100);

		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testRandOperationsInPlace() {
		if(!(cmb instanceof CompressedMatrixBlock) && rows * cols > 10000)
			return;
		RandomMatrixGenerator rgen = new RandomMatrixGenerator("uniform", rows, cols,
			ConfigurationManager.getBlocksize(), sparsity, min, max);
		Well1024a bigrand = null;
		if(!LibMatrixDatagen.isShortcutRandOperation(min, max, sparsity, RandomMatrixGenerator.PDF.UNIFORM))
			bigrand = LibMatrixDatagen.setupSeedsForRand(seed);
		MatrixBlock ret1 = cmb.randOperationsInPlace(rgen, bigrand, 1342, _k);
		if(!LibMatrixDatagen.isShortcutRandOperation(min, max, sparsity, RandomMatrixGenerator.PDF.UNIFORM))
			bigrand = LibMatrixDatagen.setupSeedsForRand(seed);
		MatrixBlock ret2 = mb.randOperationsInPlace(rgen, bigrand, 1342, _k);

		compareResultMatrices(ret1, ret2, 1);

	}

}
