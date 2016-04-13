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

package org.apache.sysml.runtime.instructions.flink.functions;

import org.apache.flink.api.common.functions.GroupCombineFunction;
import org.apache.flink.api.common.functions.JoinFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichJoinFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Collector;
import org.apache.sysml.lops.MapMultChain;
import org.apache.sysml.lops.PartialAggregate;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.functionobjects.Plus;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;

/**
 * Utility Flink functions for matrix multiplication.
 */
public final class MatrixMultiplicationFunctions {

	private MatrixMultiplicationFunctions() {
	}

	/**
	 * Join function that multiplies two matrix blocks <code>A[m,n]</code> and <code>B[s,t]</code>, assuming
	 * <code>n == s</code> and emits a new MatrixBlock <code>C[m,s] = A[m,n] * A[s,t]</code>.
	 */
	public static class MultiplyMatrixBlocks implements JoinFunction<
			Tuple2<MatrixIndexes, MatrixBlock>, Tuple2<MatrixIndexes, MatrixBlock>,
			Tuple2<MatrixIndexes, MatrixBlock>> {

		private final Tuple2<MatrixIndexes, MatrixBlock> output = new Tuple2<MatrixIndexes, MatrixBlock>();

		//created operator for reuse
		private final AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		private final AggregateBinaryOperator operation = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(),
				agg);

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> join(Tuple2<MatrixIndexes, MatrixBlock> first,
													   Tuple2<MatrixIndexes, MatrixBlock> second) throws Exception {
			if (second.f0.getRowIndex() != first.f0.getColumnIndex())
				throw new DMLRuntimeException("Dimension mismatch!");

			output.f0 = new MatrixIndexes(first.f0.getRowIndex(), second.f0.getColumnIndex());
			output.f1 = new MatrixBlock();
			first.f1.aggregateBinaryOperations(first.f1, second.f1, output.f1, operation);
			return output;
		}
	}

	/**
	 * Computes the matrix multiplication chain operation:
	 * <p>
	 * <pre><code>
	 *     t(X) %*% (X %*% v)
	 * </code></pre>
	 * <p>
	 * Where <code>v</code> is tiny and unpartitioned and broadcast to all nodes.
	 */
	public static class MultiplyTransposedMatrixBlocks extends
													   RichMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, Tuple2<MatrixIndexes, MatrixBlock>> {

		private final Tuple2<MatrixIndexes, MatrixBlock> output = new Tuple2<MatrixIndexes, MatrixBlock>();
		private final MapMultChain.ChainType type;
		private final String vName;
		private MatrixBlock v;

		public MultiplyTransposedMatrixBlocks(MapMultChain.ChainType type, String vName) {
			this.type = type;
			this.vName = vName;
			this.output.f0 = new MatrixIndexes(1, 1);
		}

		@Override
		public void open(Configuration parameters) throws Exception {
			v = getRuntimeContext().<Tuple2<?, MatrixBlock>>getBroadcastVariable(vName).get(0).f1;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> map(Tuple2<MatrixIndexes, MatrixBlock> value) throws Exception {
			if (value.f1.getNumColumns() != v.getNumRows())
				throw new DMLRuntimeException("Dimension mismatch!");
			output.f1 = value.f1.chainMatrixMultOperations(v, null, output.f1, type);
			return output;
		}
	}

	/**
	 * Computes the matrix multiplication chain operations:
	 * <p>
	 * <pre><code>
	 *     t(X) %*% (w * (X %*% v))
	 * </code></pre>
	 * <pre><code>
	 *     t(X) %*% ((X %*% v) - w)
	 * </code></pre>
	 * <p>
	 * Where <code>v</code> is tiny and unpartitioned and broadcast to all nodes, and <code>w</code> has the same
	 * partitioning as <code>X</code>.
	 */
	public static class MultiplyTransposedMatrixBlocksWithVector extends RichJoinFunction<
			Tuple2<MatrixIndexes, MatrixBlock>, Tuple2<MatrixIndexes, MatrixBlock>,
			Tuple2<MatrixIndexes, MatrixBlock>> {

		private final Tuple2<MatrixIndexes, MatrixBlock> output = new Tuple2<MatrixIndexes, MatrixBlock>();
		private final MapMultChain.ChainType type;
		private final String vName;
		private MatrixBlock v;

		public MultiplyTransposedMatrixBlocksWithVector(MapMultChain.ChainType type, String vName) {
			this.type = type;
			this.vName = vName;
			this.output.f0 = new MatrixIndexes(1, 1);
		}

		@Override
		public void open(Configuration parameters) throws Exception {
			v = getRuntimeContext().<Tuple2<?, MatrixBlock>>getBroadcastVariable(vName).get(0).f1;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> join(Tuple2<MatrixIndexes, MatrixBlock> x,
													   Tuple2<MatrixIndexes, MatrixBlock> w) throws Exception {
			if (x.f1.getNumColumns() != v.getNumRows())
				throw new DMLRuntimeException("Dimension mismatch!");
			output.f1 = x.f1.chainMatrixMultOperations(v, w.f1, output.f1, type);
			return output;
		}
	}

	/**
	 * Reduce function that sums two matrix blocks <code>C[i,j]</code> and <code>C[m,n]</code>, assuming they have the
	 * same index <code>i == m && j == n</code> and the block have the same dimensions
	 * <code>nrow(A) == nrow(B) && ncol(A) == ncol(B)</code>.
	 */
	public static class SumMatrixBlocks implements ReduceFunction<Tuple2<MatrixIndexes, MatrixBlock>> {
		private final Tuple2<MatrixIndexes, MatrixBlock> output = new Tuple2<MatrixIndexes, MatrixBlock>();
		private final BinaryOperator plus = new BinaryOperator(Plus.getPlusFnObject());

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> reduce(Tuple2<MatrixIndexes, MatrixBlock> left,
														 Tuple2<MatrixIndexes, MatrixBlock> right) throws Exception {
			if (left.f0 != right.f0)
				throw new DMLRuntimeException("Dimension mismatch!");
			output.f0 = left.f0;
			output.f1 = new MatrixBlock();
			left.f1.binaryOperations(plus, right.f1, output.f1);
			return output;
		}
	}

	/**
	 * This aggregate function uses kahan+ with corrections to aggregate input blocks; it is meant for
	 * reduce all operations where we can reuse the same correction block independent of the input
	 * block indexes. Note that this aggregation function does not apply to embedded corrections.
	 */
	public static class SumMatrixBlocksStable implements ReduceFunction<Tuple2<MatrixIndexes, MatrixBlock>> {
		private static final long serialVersionUID = 1737038715965862222L;

		private final Tuple2<MatrixIndexes, MatrixBlock> output = new Tuple2<MatrixIndexes, MatrixBlock>();

		private AggregateOperator _op = null;
		private MatrixBlock _corr = null;

		public SumMatrixBlocksStable() {
			_op = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true,
					PartialAggregate.CorrectionLocationType.NONE);
			_corr = null;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> reduce(Tuple2<MatrixIndexes, MatrixBlock> value1,
														 Tuple2<MatrixIndexes, MatrixBlock> value2)
				throws Exception {
			MatrixBlock blk1 = value1.f1;
			MatrixBlock blk2 = value2.f1;

			//create correction block (on demand)
			if (_corr == null) {
				_corr = new MatrixBlock(blk1.getNumRows(), blk1.getNumColumns(), false);
			}

			// forward martix indexes as is
			output.f0 = value1.f0;

			//copy one input to output
			output.f1 = new MatrixBlock(blk1);

			//aggregate other input
			OperationsOnMatrixValues.incrementalAggregation(output.f1, _corr, blk2, _op, false);

			return output;
		}
	}

	public static class SumMatrixBlocksCombine
			implements GroupCombineFunction<Tuple2<MatrixIndexes, MatrixBlock>, Tuple2<MatrixIndexes, MatrixBlock>> {

		private final Tuple2<MatrixIndexes, MatrixBlock> output = new Tuple2<MatrixIndexes, MatrixBlock>();
		private final BinaryOperator plus = new BinaryOperator(Plus.getPlusFnObject());

		@Override
		public void combine(Iterable<Tuple2<MatrixIndexes, MatrixBlock>> values,
							Collector<Tuple2<MatrixIndexes, MatrixBlock>> out) throws Exception {
			output.f0 = null;
			output.f1 = null;
			for (Tuple2<MatrixIndexes, MatrixBlock> val : values) {
				if (output.f0 == null) {
					output.f0 = val.f0;
					output.f1 = val.f1;
				} else {
					output.f1 = (MatrixBlock) output.f1.binaryOperations(plus, val.f1, new MatrixBlock());
				}
			}
			out.collect(output);
		}
	}

	/**
	 * {@link KeySelector} implementation that retrieves the row index of a matrix block.
	 */
	public static class RowSelector implements KeySelector<Tuple2<MatrixIndexes, MatrixBlock>, Long> {

		@Override
		public Long getKey(Tuple2<MatrixIndexes, MatrixBlock> value) throws Exception {
			return value.f0.getRowIndex();
		}
	}

	/**
	 * {@link KeySelector} implementation that retrieves the column index of a matrix block.
	 */
	public static class ColumnSelector implements KeySelector<Tuple2<MatrixIndexes, MatrixBlock>, Long> {

		@Override
		public Long getKey(Tuple2<MatrixIndexes, MatrixBlock> value) throws Exception {
			return value.f0.getColumnIndex();
		}
	}

	/**
	 * {@link KeySelector} implementation that retrieves indexes of a matrix block.
	 */
	public static class MatrixIndexesSelector
			implements KeySelector<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes> {

		@Override
		public MatrixIndexes getKey(Tuple2<MatrixIndexes, MatrixBlock> value) throws Exception {
			return value.f0;
		}
	}
}
