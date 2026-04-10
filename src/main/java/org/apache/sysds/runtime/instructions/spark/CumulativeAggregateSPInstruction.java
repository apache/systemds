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

package org.apache.sysds.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.PlusMultiply;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import scala.Tuple2;

public class CumulativeAggregateSPInstruction extends AggregateUnarySPInstruction {

	private CumulativeAggregateSPInstruction(AggregateUnaryOperator op, CPOperand in1, CPOperand out, String opcode, String istr) {
		super(SPType.CumsumAggregate, op, null, in1, out, null, opcode, istr);
	}

	public static CumulativeAggregateSPInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 2);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		AggregateUnaryOperator aggun = InstructionUtils.parseCumulativeAggregateUnaryOperator(opcode);
		return new CumulativeAggregateSPInstruction(aggun, in1, out, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext) ec;
		DataCharacteristics mc = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics mcOut = new MatrixCharacteristics(mc);

		long rlen = mc.getRows();
		long clen = mc.getCols();
		int blen = mc.getBlocksize();

		// Row-cumsum in Spark reuses the current cumulative skeleton, but the
		// partial aggregates are compressed across column blocks instead of row blocks.
		// We infer this orientation from the already planned output dimensions.
		DataCharacteristics plannedOut = sec.getDataCharacteristics(output.getName());
		boolean rowCum = plannedOut != null
				&& plannedOut.getRows() == rlen
				&& plannedOut.getCols() > 0
				&& clen > 0
				&& plannedOut.getCols() == (long) Math.ceil((double) clen / blen);

		if (rowCum)
			mcOut.set(rlen, (long) Math.ceil((double) clen / blen), blen, -1);
		else
			mcOut.set((long) Math.ceil((double) rlen / blen), clen, blen, -1);

		JavaPairRDD<MatrixIndexes, MatrixBlock> in =
				sec.getBinaryMatrixBlockRDDHandleForVariable(input1.getName());

		AggregateUnaryOperator auop = (AggregateUnaryOperator) _optr;
		JavaPairRDD<MatrixIndexes, MatrixBlock> out =
				in.mapToPair(new RDDCumAggFunction(auop, rlen, clen, blen, rowCum));

		int numParts = SparkUtils.getNumPreferredPartitions(mcOut);
		int minPar = (int) Math.min(
				SparkExecutionContext.getDefaultParallelism(true),
				mcOut.getNumBlocks());

		out = RDDAggregateUtils.mergeByKey(out, Math.max(numParts, minPar), false);

		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		sec.getDataCharacteristics(output.getName()).set(mcOut);
	}

	private static class RDDCumAggFunction
			implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> {
		private static final long serialVersionUID = 11324676268945117L;

		private final AggregateUnaryOperator _op;
		private UnaryOperator _uop = null;
		private final long _rlen;
		private final long _clen;
		private final int _blen;
		private final boolean _rowCum;

		public RDDCumAggFunction(AggregateUnaryOperator op, long rlen, long clen, int blen, boolean rowCum) {
			_op = op;
			_rlen = rlen;
			_clen = clen;
			_blen = blen;
			_rowCum = rowCum;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> arg0)
				throws Exception {
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();

			MatrixIndexes ixOut = new MatrixIndexes();
			MatrixBlock blkOut = new MatrixBlock();

			AggregateUnaryOperator aop = _op;

			if (aop.aggOp.increOp.fn instanceof PlusMultiply) { // cumsumprod
				aop.indexFn.execute(ixIn, ixOut);
				if (_uop == null)
					_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucumk+*"));
				MatrixBlock t1 = blkIn.unaryOperations(_uop, new MatrixBlock());
				MatrixBlock t2 = blkIn.slice(0, blkIn.getNumRows() - 1, 1, 1, new MatrixBlock());
				blkOut.reset(1, 2);
				blkOut.set(0, 0, t1.get(t1.getNumRows() - 1, 0));
				blkOut.set(0, 1, t2.prod());
			}
			else if (_rowCum) {
				// For Spark rowcumsum, the carry across column blocks is the row-wise sum
				// of each input block. This yields a (rowsInBlock x 1) partial aggregate.
				int nr = blkIn.getNumRows();
				int nc = blkIn.getNumColumns();
				blkOut.reset(nr, 1, false);

				for (int i = 0; i < nr; i++) {
					double sum = 0;
					for (int j = 0; j < nc; j++)
						sum += blkIn.get(i, j);
					blkOut.set(i, 0, sum);
				}

				ixOut.setIndexes(ixIn.getRowIndex(), ixIn.getColumnIndex());
			}
			else {
				OperationsOnMatrixValues.performAggregateUnary(ixIn, blkIn, ixOut, blkOut, aop, _blen);
				if (aop.aggOp.existsCorrection())
					blkOut.dropLastRowsOrColumns(aop.aggOp.correction);
			}

			if (!_rowCum) {
				long rlenOut = (long) Math.ceil((double) _rlen / _blen);
				long rixOut = (long) Math.ceil((double) ixIn.getRowIndex() / _blen);
				int rlenBlk = (int) Math.min(rlenOut - (rixOut - 1) * _blen, _blen);
				int clenBlk = blkOut.getNumColumns();
				int posBlk = (int) ((ixIn.getRowIndex() - 1) % _blen);

				MatrixBlock blkOut2 = new MatrixBlock(rlenBlk, clenBlk, true);
				blkOut2.copy(posBlk, posBlk, 0, clenBlk - 1, blkOut, true);
				ixOut.setIndexes(rixOut, ixOut.getColumnIndex());

				return new Tuple2<>(ixOut, blkOut2);
			}
			else {
				long clenOut = (long) Math.ceil((double) _clen / _blen);
				long cixOut = (long) Math.ceil((double) ixIn.getColumnIndex() / _blen);
				int rlenBlk = blkOut.getNumRows();
				int clenBlk = (int) Math.min(clenOut - (cixOut - 1) * _blen, _blen);
				int posBlk = (int) ((ixIn.getColumnIndex() - 1) % _blen);

				MatrixBlock blkOut2 = new MatrixBlock(rlenBlk, clenBlk, true);
				blkOut2.copy(0, rlenBlk - 1, posBlk, posBlk, blkOut, true);
				ixOut.setIndexes(ixIn.getRowIndex(), cixOut);

				return new Tuple2<>(ixOut, blkOut2);
			}
		}
	}
}