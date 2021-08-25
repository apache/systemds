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

import static org.junit.Assert.assertTrue;

import java.util.Collection;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.TestConstants.MatrixTypology;
import org.apache.sysds.test.component.compress.TestConstants.OverLapping;
import org.apache.sysds.test.component.compress.TestConstants.SparsityType;
import org.apache.sysds.test.component.compress.TestConstants.ValueRange;
import org.apache.sysds.test.component.compress.TestConstants.ValueType;
import org.junit.Test;

public abstract class AbstractCompressedUnaryTests extends CompressedTestBase {

	public AbstractCompressedUnaryTests(SparsityType sparType, ValueType valType, ValueRange valRange,
		CompressionSettingsBuilder compSettings, MatrixTypology matrixTypology, OverLapping ov, int parallelism,
		Collection<CompressionType> ct) {
		super(sparType, valType, valRange, compSettings, matrixTypology, ov, parallelism, ct);
	}

	enum AggType {
		ROWSUMS, COLSUMS, SUM, ROWSUMSSQ, COLSUMSSQ, SUMSQ, ROWMAXS, COLMAXS, MAX, ROWMINS, COLMINS, MIN, MEAN, COLMEAN,
		ROWMEAN
	}

	@Test
	public void testUnaryOperator_ROWSUMS_CP() {
		testUnaryOperators(AggType.ROWSUMS, true);
	}

	@Test
	public void testUnaryOperator_COLSUMS_CP() {
		testUnaryOperators(AggType.COLSUMS, true);
	}

	@Test
	public void testUnaryOperator_SUM_CP() {
		testUnaryOperators(AggType.SUM, true);
	}

	@Test
	public void testUnaryOperator_ROWSUMSSQ_CP() {
		testUnaryOperators(AggType.ROWSUMSSQ, true);
	}

	@Test
	public void testUnaryOperator_COLSUMSSQ_CP() {
		testUnaryOperators(AggType.COLSUMSSQ, true);
	}

	@Test
	public void testUnaryOperator_SUMSQ_CP() {
		testUnaryOperators(AggType.SUMSQ, true);
	}

	@Test
	public void testUnaryOperator_ROWMAXS_CP() {
		testUnaryOperators(AggType.ROWMAXS, true);
	}

	@Test
	public void testUnaryOperator_COLMAXS_CP() {
		testUnaryOperators(AggType.COLMAXS, true);
	}

	@Test
	public void testUnaryOperator_MAX_CP() {
		testUnaryOperators(AggType.MAX, true);
	}

	@Test
	public void testUnaryOperator_ROWMINS_CP() {
		testUnaryOperators(AggType.ROWMINS, true);
	}

	@Test
	public void testUnaryOperator_COLMINS_CP() {
		testUnaryOperators(AggType.COLMINS, true);
	}

	@Test
	public void testUnaryOperator_MIN_CP() {
		testUnaryOperators(AggType.MIN, true);
	}

	@Test
	public void testUnaryOperator_MEAN_CP() {
		testUnaryOperators(AggType.MEAN, true);
	}

	@Test
	public void testUnaryOperator_COLMEAN_CP() {
		testUnaryOperators(AggType.COLMEAN, true);
	}

	@Test
	public void testUnaryOperator_ROWMEAN_CP() {
		testUnaryOperators(AggType.ROWMEAN, true);
	}

	@Test
	public void testUnaryOperator_ROWSUMS_SP() {
		testUnaryOperators(AggType.ROWSUMS, false);
	}

	@Test
	public void testUnaryOperator_COLSUMS_SP() {
		testUnaryOperators(AggType.COLSUMS, false);
	}

	@Test
	public void testUnaryOperator_SUM_SP() {
		testUnaryOperators(AggType.SUM, false);
	}

	@Test
	public void testUnaryOperator_ROWSUMSSQ_SP() {
		testUnaryOperators(AggType.ROWSUMSSQ, false);
	}

	@Test
	public void testUnaryOperator_COLSUMSSQ_SP() {
		testUnaryOperators(AggType.COLSUMSSQ, false);
	}

	@Test
	public void testUnaryOperator_SUMSQ_SP() {
		testUnaryOperators(AggType.SUMSQ, false);
	}

	@Test
	public void testUnaryOperator_ROWMAXS_SP() {
		testUnaryOperators(AggType.ROWMAXS, false);
	}

	@Test
	public void testUnaryOperator_COLMAXS_SP() {
		testUnaryOperators(AggType.COLMAXS, false);
	}

	@Test
	public void testUnaryOperator_MAX_SP() {
		testUnaryOperators(AggType.MAX, false);
	}

	@Test
	public void testUnaryOperator_ROWMINS_SP() {
		testUnaryOperators(AggType.ROWMINS, false);
	}

	@Test
	public void testUnaryOperator_COLMINS_SP() {
		testUnaryOperators(AggType.COLMINS, false);
	}

	@Test
	public void testUnaryOperator_MIN_SP() {
		testUnaryOperators(AggType.MIN, false);
	}

	@Test
	public void testUnaryOperator_MEAN_SP() {
		testUnaryOperators(AggType.MEAN, false);
	}

	@Test
	public void testUnaryOperator_COLMEAN_SP() {
		testUnaryOperators(AggType.COLMEAN, false);
	}

	@Test
	public void testUnaryOperator_ROWMEAN_SP() {
		testUnaryOperators(AggType.ROWMEAN, false);
	}

	protected AggregateUnaryOperator getUnaryOperator(AggType aggType, int threads) {
		switch(aggType) {
			case SUM:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uak+", threads);
			case ROWSUMS:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uark+", threads);
			case COLSUMS:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uack+", threads);
			case SUMSQ:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uasqk+", threads);
			case ROWSUMSSQ:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uarsqk+", threads);
			case COLSUMSSQ:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uacsqk+", threads);
			case MAX:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uamax", threads);
			case ROWMAXS:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uarmax", threads);
			case COLMAXS:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uacmax", threads);
			case MIN:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uamin", threads);
			case ROWMINS:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uarmin", threads);
			case COLMINS:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uacmin", threads);
			case MEAN:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uamean", threads);
			case ROWMEAN:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uarmean", threads);
			case COLMEAN:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uacmean", threads);
			default:
				throw new NotImplementedException("Not Supported Aggregate Unary operator in test");
		}
	}

	public abstract void testUnaryOperators(AggType aggType, boolean inCP);

	public void testUnaryOperators(AggType aggType, AggregateUnaryOperator auop, boolean inCP) {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test
			// matrix-vector uncompressed
			MatrixBlock ret1 = mb.aggregateUnaryOperations(auop, new MatrixBlock(), Math.max(rows, cols), null, inCP);
			// matrix-vector compressed
			MatrixBlock ret2 = cmb.aggregateUnaryOperations(auop, new MatrixBlock(), Math.max(rows, cols), null, inCP);
			// LOG.error(ret1 + "\nvs\n" + ret2);
			// LOG.error(cmb);
			// compare result with input

			assertTrue("dim 1 is not equal in compressed res  should be : " + ret1.getNumRows() + "  but is: "
				+ ret2.getNumRows(), ret1.getNumRows() == ret2.getNumRows());

			assertTrue("dim 2 is not equal in compressed res  should be : " + ret1.getNumColumns() + "  but is: "
				+ ret2.getNumColumns(), ret1.getNumColumns() == ret2.getNumColumns());
			if(!inCP){
				ret1.dropLastRowsOrColumns(auop.aggOp.correction);
				ret2.dropLastRowsOrColumns(auop.aggOp.correction);
			}

			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);

			String css = this.toString();
			if(_cs != null && _cs.lossy) {
				if(aggType == AggType.COLSUMS)
					TestUtils.compareMatrices(d1, d2, lossyTolerance * 10 * rows, css);
				else if(aggType == AggType.ROWSUMS)
					TestUtils.compareMatrices(d1, d2, lossyTolerance * 16 * cols, css);
				else if(aggType == AggType.ROWSUMSSQ)
					TestUtils.compareMatricesPercentageDistance(d1, d2, 0.5, 0.9, css, true);
				else if(aggType == AggType.SUM)
					TestUtils.compareMatrices(d1, d2, lossyTolerance * 10 * cols * rows, css);
				else if(aggType == AggType.MEAN)
					TestUtils.compareMatrices(d1, d2, lossyTolerance * cols * rows, css);
				else if(aggType == AggType.ROWMEAN)
					TestUtils.compareMatrices(d1, d2, lossyTolerance, css);
				else
					TestUtils.compareMatricesPercentageDistance(d1, d2, 0.8, 0.9, css, true);

			}
			else{
				if(overlappingType == OverLapping.SQUASH) 
					TestUtils.compareMatricesPercentageDistance(d1, d2, 0.0, 0.90, css);
				else if(aggType == AggType.ROWMEAN)
					TestUtils.compareMatrices(d1, d2, 0.0001, css);
				else if(OverLapping.effectOnOutput(overlappingType))
					TestUtils.compareMatricesPercentageDistance(d1, d2, 0.95, 0.98, css);
				else
					TestUtils.compareMatricesBitAvgDistance(d1, d2, 2048, 128, css);
			}

		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}
}
