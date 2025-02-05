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

import static org.junit.Assert.fail;

import java.util.Collection;
import java.util.concurrent.atomic.AtomicBoolean;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.cost.CostEstimatorBuilder;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
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
		Collection<CompressionType> ct, CostEstimatorBuilder csb) {
		super(sparType, valType, valRange, compSettings, matrixTypology, ov, parallelism, ct, csb);
	}

	enum AggType {
		ROW_SUMS, COL_SUMS, SUM, ROW_SUMS_SQ, COL_SUMS_SQ, SUM_SQ, ROW_MAXS, COL_MAXS, MAX, ROW_MINS, COL_MINS, MIN, MEAN,
		COL_MEAN, ROW_MEAN, PRODUCT
	}

	@Test
	public void testUnaryOperator_ROW_SUMS_CP() {
		testUnaryOperators(AggType.ROW_SUMS, true);
	}

	@Test
	public void testUnaryOperator_COL_SMS_CP() {
		testUnaryOperators(AggType.COL_SUMS, true);
	}

	@Test
	public void testUnaryOperator_SUM_CP() {
		testUnaryOperators(AggType.SUM, true);
	}

	@Test
	public void testUnaryOperator_ROW_SUMS_SQ_CP() {
		testUnaryOperators(AggType.ROW_SUMS_SQ, true);
	}

	@Test
	public void testUnaryOperator_COL_SUMS_SQ_CP() {
		testUnaryOperators(AggType.COL_SUMS_SQ, true);
	}

	@Test
	public void testUnaryOperator_SUM_SQ_CP() {
		testUnaryOperators(AggType.SUM_SQ, true);
	}

	@Test
	public void testUnaryOperator_ROW_MAXS_CP() {
		testUnaryOperators(AggType.ROW_MAXS, true);
	}

	@Test
	public void testUnaryOperator_COL_MAXS_CP() {
		testUnaryOperators(AggType.COL_MAXS, true);
	}

	@Test
	public void testUnaryOperator_MAX_CP() {
		testUnaryOperators(AggType.MAX, true);
	}

	@Test
	public void testUnaryOperator_ROW_MINS_CP() {
		testUnaryOperators(AggType.ROW_MINS, true);
	}

	@Test
	public void testUnaryOperator_COL_MINS_CP() {
		testUnaryOperators(AggType.COL_MINS, true);
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
	public void testUnaryOperator_COL_MEAN_CP() {
		testUnaryOperators(AggType.COL_MEAN, true);
	}

	@Test
	public void testUnaryOperator_ROW_MEAN_CP() {
		testUnaryOperators(AggType.ROW_MEAN, true);
	}

	@Test
	public void testUnaryOperator_ROW_SUMS_SP() {
		testUnaryOperators(AggType.ROW_SUMS, false);
	}

	@Test
	public void testUnaryOperator_COL_SUMS_SP() {
		testUnaryOperators(AggType.COL_SUMS, false);
	}

	@Test
	public void testUnaryOperator_SUM_SP() {
		testUnaryOperators(AggType.SUM, false);
	}

	@Test
	public void testUnaryOperator_ROW_SUMS_SQ_SP() {
		testUnaryOperators(AggType.ROW_SUMS_SQ, false);
	}

	@Test
	public void testUnaryOperator_COL_SUMS_SQ_SP() {
		testUnaryOperators(AggType.COL_SUMS_SQ, false);
	}

	@Test
	public void testUnaryOperator_SUM_SQ_SP() {
		testUnaryOperators(AggType.SUM_SQ, false);
	}

	@Test
	public void testUnaryOperator_ROW_MAXS_SP() {
		testUnaryOperators(AggType.ROW_MAXS, false);
	}

	@Test
	public void testUnaryOperator_COL_MAXS_SP() {
		testUnaryOperators(AggType.COL_MAXS, false);
	}

	@Test
	public void testUnaryOperator_MAX_SP() {
		testUnaryOperators(AggType.MAX, false);
	}

	@Test
	public void testUnaryOperator_ROW_MINS_SP() {
		testUnaryOperators(AggType.ROW_MINS, false);
	}

	@Test
	public void testUnaryOperator_COL_MINS_SP() {
		testUnaryOperators(AggType.COL_MINS, false);
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
	public void testUnaryOperator_COL_MEAN_SP() {
		testUnaryOperators(AggType.COL_MEAN, false);
	}

	@Test
	public void testUnaryOperator_ROW_MEAN_SP() {
		testUnaryOperators(AggType.ROW_MEAN, false);
	}

	@Test
	public void testUnaryOperator_PROD_CP() {
		testUnaryOperators(AggType.PRODUCT, true);
	}

	@Test
	public void testUnaryOperator_PROD_SP() {
		testUnaryOperators(AggType.PRODUCT, false);
	}

	protected AggregateUnaryOperator getUnaryOperator(AggType aggType, int threads) {
		switch(aggType) {
			case SUM:
				return InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAKP.toString(), threads);
			case ROW_SUMS:
				return InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARKP.toString(), threads);
			case COL_SUMS:
				return InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UACKP.toString(), threads);
			case SUM_SQ:
				return InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UASQKP.toString(), threads);
			case ROW_SUMS_SQ:
				return InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARSQKP.toString(), threads);
			case COL_SUMS_SQ:
				return InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UACSQKP.toString(), threads);
			case MAX:
				return InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAMAX.toString(), threads);
			case ROW_MAXS:
				return InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARMAX.toString(), threads);
			case COL_MAXS:
				return InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UACMAX.toString(), threads);
			case MIN:
				return InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAMIN.toString(), threads);
			case ROW_MINS:
				return InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARMIN.toString(), threads);
			case COL_MINS:
				return InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UACMIN.toString(), threads);
			case MEAN:
				return InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAMEAN.toString(), threads);
			case ROW_MEAN:
				return InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARMEAN.toString(), threads);
			case COL_MEAN:
				return InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UACMEAN.toString(), threads);
			case PRODUCT:
				return InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAM.toString(), threads);
			default:
				throw new NotImplementedException("Not Supported Aggregate Unary operator in test");
		}
	}

	public abstract void testUnaryOperators(AggType aggType, boolean inCP);

	private static AtomicBoolean printedWarningForProduct = new AtomicBoolean(false);

	public void testUnaryOperators(AggType aggType, AggregateUnaryOperator auop, boolean inCP) {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			// matrix-vector uncompressed
			MatrixBlock ret1 = mb.aggregateUnaryOperations(auop, new MatrixBlock(), Math.max(rows, cols), null, inCP);
			// matrix-vector compressed
			MatrixBlock ret2 = cmb.aggregateUnaryOperations(auop, new MatrixBlock(), Math.max(rows, cols), null, inCP);

			final int ruc = ret1.getNumRows();
			final int cuc = ret1.getNumColumns();
			final int rc = ret2.getNumRows();
			final int cc = ret2.getNumColumns();
			if(ruc != rc)
				fail("dim 1 is not equal in compressed res  should be : " + ruc + "  but is: " + rc);
			if(cuc != cc)
				fail("dim 2 is not equal in compressed res  should be : " + rc + "  but is: " + cc);

			if(!inCP) {
				ret1.dropLastRowsOrColumns(auop.aggOp.correction);
				ret2.dropLastRowsOrColumns(auop.aggOp.correction);
			}

			String css = this.toString();
			if(_cs != null && _cs.lossy) {
				if(aggType == AggType.COL_SUMS)
					TestUtils.compareMatrices(ret1, ret2, lossyTolerance * 10 * rows, css);
				else if(aggType == AggType.ROW_SUMS)
					TestUtils.compareMatrices(ret1, ret2, lossyTolerance * 16 * cols, css);
				else if(aggType == AggType.ROW_SUMS_SQ)
					TestUtils.compareMatricesPercentageDistance(ret1, ret2, 0.5, 0.9, css, true);
				else if(aggType == AggType.SUM)
					TestUtils.compareMatrices(ret1, ret2, lossyTolerance * 10 * cols * rows, css);
				else if(aggType == AggType.MEAN)
					TestUtils.compareMatrices(ret1, ret2, lossyTolerance * cols * rows, css);
				else if(aggType == AggType.ROW_MEAN)
					TestUtils.compareMatrices(ret1, ret2, lossyTolerance, css);
				else
					TestUtils.compareMatricesPercentageDistance(ret1, ret2, 0.8, 0.9, css, true);
			}
			else {
				if(aggType == AggType.PRODUCT) {
					if(Double.isInfinite(ret2.get(0, 0))){
						if(!printedWarningForProduct.get()) {
							printedWarningForProduct.set(true);
							LOG.warn("Product not equal because of double rounding upwards");
						}
						return;
					}
					if(Math.abs(ret2.get(0, 0)) <= 1E-16) {
						if(!printedWarningForProduct.get()) {
							printedWarningForProduct.set(true);
							LOG.warn("Product not equal because of  double rounding downwards");
						}

						return; // Product is weird around zero.
					}
					TestUtils.compareMatricesPercentageDistance(ret1, ret2, 0.95, 0.98, css);
				}
				else if(overlappingType == OverLapping.SQUASH)
					TestUtils.compareMatricesPercentageDistance(ret1, ret2, 0.0, 0.90, css);
				else if(aggType == AggType.ROW_MEAN)
					TestUtils.compareMatrices(ret1, ret2, 0.0001, css);
				else if(OverLapping.effectOnOutput(overlappingType))
					TestUtils.compareMatricesPercentageDistance(ret1, ret2, 0.95, 0.98, css);
				else
					TestUtils.compareMatricesBitAvgDistance(ret1, ret2, 2048, 512, css);

			}

		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}
}
