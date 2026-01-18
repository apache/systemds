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
		// parts: opcode, in, out  => 3 fields
		InstructionUtils.checkNumFields(parts, 3);

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

		// get input
		JavaPairRDD<MatrixIndexes, MatrixBlock> in =
				sec.getBinaryMatrixBlockRDDHandleForVariable(input1.getName());

		if ("urowcumk+".equals(getOpcode())) {
			// rowcumsum: aggregate phase should output carry/end-values per block
			processRowCumsumAggregate(sec, in, mc);
		}
		else {
			// regular cumsum aggregate phase
			processCumsum(sec, in, mc);
		}
	}

	private void processRowCumsumAggregate(SparkExecutionContext sec, JavaPairRDD<MatrixIndexes, MatrixBlock> in, DataCharacteristics mc) {
		Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, JavaPairRDD<MatrixIndexes, MatrixBlock>> res =
				processRowCumsumWithEndValues(in);

		JavaPairRDD<MatrixIndexes, MatrixBlock> endValues = res._2;

		sec.setRDDHandleForVariable(output.getName(), endValues);
		sec.addLineageRDD(output.getName(), input1.getName());

		// output characteristics: same rows as input, but 1 column (per-row carry)
		MatrixCharacteristics mcOut = new MatrixCharacteristics(mc);
		mcOut.setCols(1);
		sec.getDataCharacteristics(output.getName()).set(mcOut);
	}

	/**
	 * Helper for rowcumsum:
	 * returns (localRowCumsumBlocks, endValuesBlocks).
	 */
	public static Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, JavaPairRDD<MatrixIndexes, MatrixBlock>>
	processRowCumsumWithEndValues(JavaPairRDD<MatrixIndexes, MatrixBlock> in) {

		JavaPairRDD<MatrixIndexes, MatrixBlock> localRowCumsum =
				in.mapToPair(new LocalRowCumsumFunction());

		JavaPairRDD<MatrixIndexes, MatrixBlock> endValues =
				localRowCumsum.mapToPair(new ExtractEndValuesFunction());

		return new Tuple2<>(localRowCumsum, endValues);
	}

	/**
	 * Original cumsum aggregate phase (keep intact).
	 */
	private void processCumsum(SparkExecutionContext sec, JavaPairRDD<MatrixIndexes, MatrixBlock> in, DataCharacteristics mc) {
		DataCharacteristics mcOut = new MatrixCharacteristics(mc);
		long rlen = mc.getRows();
		int blen = mc.getBlocksize();
		mcOut.setRows((long) (Math.ceil((double) rlen / blen)));

		// execute unary aggregate (w/ implicit drop correction)
		AggregateUnaryOperator auop = (AggregateUnaryOperator) _optr;
		JavaPairRDD<MatrixIndexes, MatrixBlock> out =
				in.mapToPair(new RDDCumAggFunction(auop, rlen, blen));

		// merge partial aggregates, adjusting for correct number of partitions
		int numParts = SparkUtils.getNumPreferredPartitions(mcOut);
		int minPar = (int) Math.min(SparkExecutionContext.getDefaultParallelism(true), mcOut.getNumBlocks());
		out = RDDAggregateUtils.mergeByKey(out, Math.max(numParts, minPar), false);

		// put output handle in symbol table
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		sec.getDataCharacteristics(output.getName()).set(mcOut);
	}


	private static class LocalRowCumsumFunction
			implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> {

		private static final long serialVersionUID = 123L;

		private static final UnaryOperator ROWCUMSUM_OP =
				new UnaryOperator(Builtin.getBuiltinFnObject("urowcumk+"));

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> kv) {
			MatrixIndexes idx = kv._1;
			MatrixBlock inputBlock = kv._2;

			MatrixBlock outBlock = inputBlock.unaryOperations(ROWCUMSUM_OP, new MatrixBlock());
			return new Tuple2<>(idx, outBlock);
		}
	}


	private static class ExtractEndValuesFunction
			implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> {

		private static final long serialVersionUID = 123L;

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> kv) {
			MatrixIndexes idx = kv._1;
			MatrixBlock cumsumBlock = kv._2;

			int r = cumsumBlock.getNumRows();
			int c = cumsumBlock.getNumColumns();
			MatrixBlock endValuesBlock = new MatrixBlock(r, 1, false);

			if (c > 0) {
				int lastCol = c - 1;
				for (int i = 0; i < r; i++) {
					endValuesBlock.set(i, 0, cumsumBlock.get(i, lastCol));
				}
			}
			else {
				// degenerate case: empty block
				for (int i = 0; i < r; i++) {
					endValuesBlock.set(i, 0, 0.0);
				}
			}

			return new Tuple2<>(idx, endValuesBlock);
		}
	}

	private static class RDDCumAggFunction
			implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> {

		private static final long serialVersionUID = 11324676268945117L;

		private final AggregateUnaryOperator _op;
		private UnaryOperator _uop = null;
		private final long _rlen;
		private final int _blen;

		public RDDCumAggFunction(AggregateUnaryOperator op, long rlen, int blen) {
			_op = op;
			_rlen = rlen;
			_blen = blen;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) throws Exception {
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();

			MatrixIndexes ixOut = new MatrixIndexes();
			MatrixBlock blkOut = new MatrixBlock();

			// process instruction
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
			else { // general case
				OperationsOnMatrixValues.performAggregateUnary(ixIn, blkIn, ixOut, blkOut, aop, _blen);
				if (aop.aggOp.existsCorrection())
					blkOut.dropLastRowsOrColumns(aop.aggOp.correction);
			}

			// cumsum expand partial aggregates
			long rlenOut = (long) Math.ceil((double) _rlen / _blen);
			long rixOut = (long) Math.ceil((double) ixIn.getRowIndex() / _blen);
			int rlenBlk = (int) Math.min(rlenOut - (rixOut - 1) * _blen, _blen);
			int clenBlk = blkOut.getNumColumns();
			int posBlk = (int) ((ixIn.getRowIndex() - 1) % _blen);

			// construct sparse output blocks (single row in target block size)
			MatrixBlock blkOut2 = new MatrixBlock(rlenBlk, clenBlk, true);
			blkOut2.copy(posBlk, posBlk, 0, clenBlk - 1, blkOut, true);
			ixOut.setIndexes(rixOut, ixOut.getColumnIndex());

			return new Tuple2<>(ixOut, blkOut2);
		}
	}
}
