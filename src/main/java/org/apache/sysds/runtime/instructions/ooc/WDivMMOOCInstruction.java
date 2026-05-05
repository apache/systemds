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

package org.apache.sysds.runtime.instructions.ooc;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.lops.WeightedDivMM.WDivMMType;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.QuaternaryOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;

import java.util.function.Function;

public class WDivMMOOCInstruction extends QuaternaryOOCInstruction {

	protected WDivMMOOCInstruction(QuaternaryOperator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4,
		CPOperand out, String opcode, String istr) {
		super(op, in1, in2, in3, in4, out, opcode, istr);
	}

	public static WDivMMOOCInstruction parseInstruction(QuaternaryOOCInstruction instr) {
		String instrStr = instr.getInstructionString();
		String opcode = InstructionUtils.getInstructionPartsWithValueType(instr.getInstructionString())[0];
		return new WDivMMOOCInstruction((QuaternaryOperator) instr.getOperator(), instr.input1, instr.input2,
			instr.input3, instr.input4, instr.output, opcode, instrStr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		QuaternaryOperator qop = ((QuaternaryOperator) _optr);
		final WDivMMType wt = qop.wtype3;

		CachingStream X = new CachingStream(ec.getMatrixObject(input1).getStreamHandle());
		CachingStream U = new CachingStream(ec.getMatrixObject(input2).getStreamHandle());
		CachingStream V = new CachingStream(ec.getMatrixObject(input3).getStreamHandle());

		boolean basic = wt.isBasic();
		boolean left = wt.isLeft();
		boolean mult = wt.isMult();
		boolean minus = wt.isMinus();
		boolean four = wt.hasFourInputs();
		boolean scalar = wt.hasScalar();

		OOCStream<IndexedMatrixValue> mmt = matMultOOC(U.getReadStream(), V.getReadStream(), U.getDataCharacteristics(),
			V.getDataCharacteristics(), false, true);
		OOCStream<IndexedMatrixValue> inter;
		OOCStream<IndexedMatrixValue> out;

		if(basic) {
			out = elemMultOOC(X.getReadStream(), mmt);
			ec.getMatrixObject(output).setStreamHandle(out);
			return;
		}
		else if(four) {
			if(scalar) {
				double eps = ec.getScalarInput(input4).getDoubleValue();
				inter = elemDivOOC(X.getReadStream(), elemPlusOOC(mmt, eps));
			}
			else {
				CachingStream W = new CachingStream(ec.getMatrixObject(input4).getStreamHandle());
				inter = elemMultOOC(X.getReadStream(), elemMinusOOC(mmt, W.getReadStream()));
			}
		}
		else {
			if(minus)
				inter = maskOOC(X.getReadStream(), elemMinusOOC(mmt, X.getReadStream()));
			else {
				if(mult)
					inter = elemMultOOC(X.getReadStream(), mmt);
				else
					inter = elemDivOOC(X.getReadStream(), mmt);
			}
		}

		if(left)
			out = matMultOOC(inter, U.getReadStream(), X.getDataCharacteristics(), U.getDataCharacteristics(),
				true, false);
		else
			out = matMultOOC(inter, V.getReadStream(), X.getDataCharacteristics(), V.getDataCharacteristics(),
				false, false);

		ec.getMatrixObject(output).setStreamHandle(out);
	}

	private OOCStream<IndexedMatrixValue> matMultOOC(OOCStream<IndexedMatrixValue> m1, OOCStream<IndexedMatrixValue> m2,
		DataCharacteristics dc1, DataCharacteristics dc2, boolean leftTranspose, boolean rightTranspose) {

		int emitLeftThreshold = rightTranspose ? (int) dc2.getNumRowBlocks() : (int) dc2.getNumColBlocks();
		int emitRightThreshold = leftTranspose ? (int) dc1.getNumColBlocks() : (int) dc1.getNumRowBlocks();

		OOCStream<IndexedMatrixValue> intermediateStream = createWritableStream();
		OOCStream<IndexedMatrixValue> out = createWritableStream();

		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator op = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);

		joinManyOOC(m1, m2, intermediateStream, (left, right) -> {
			MatrixBlock leftBlock = (MatrixBlock) left.getValue();
			MatrixBlock rightBlock = (MatrixBlock) right.getValue();
			if(leftTranspose)
				leftBlock = leftBlock.transpose();
			if(rightTranspose)
				rightBlock = rightBlock.transpose();

			MatrixBlock partialResult = leftBlock.aggregateBinaryOperations(leftBlock, rightBlock, new MatrixBlock(), op);
			int lidx = (int) (leftTranspose ? left.getIndexes().getColumnIndex() : left.getIndexes().getRowIndex());
			int ridx = (int) (rightTranspose ? right.getIndexes().getRowIndex() : right.getIndexes().getColumnIndex());
			return new IndexedMatrixValue(new MatrixIndexes(lidx, ridx), partialResult);
		}, tmp -> leftTranspose ? tmp.getIndexes().getRowIndex() : tmp.getIndexes().getColumnIndex(),
			tmp -> rightTranspose ? tmp.getIndexes().getColumnIndex() : tmp.getIndexes().getRowIndex(),
			emitLeftThreshold, emitRightThreshold);

		BinaryOperator plus = InstructionUtils.parseBinaryOperator(Opcodes.PLUS.toString());
		int emitAggThreshold = leftTranspose ? (int) dc1.getNumRowBlocks() : (int) dc1.getNumColBlocks();

		groupedReduceOOC(intermediateStream, out, (left, right) -> {
			MatrixBlock mb = ((MatrixBlock) left.getValue()).binaryOperationsInPlace(plus, right.getValue());
			left.setValue(mb);
			return left;
		}, emitAggThreshold);

		return out;
	}

	private OOCStream<IndexedMatrixValue> elemOOC(OOCStream<IndexedMatrixValue> m1, OOCStream<IndexedMatrixValue> m2, BinaryOperator bop) {
		SubscribableTaskQueue<IndexedMatrixValue> out = new SubscribableTaskQueue<>();
		Function<IndexedMatrixValue, MatrixIndexes> key = imv ->
			new MatrixIndexes(imv.getIndexes().getRowIndex(), imv.getIndexes().getColumnIndex());

		joinOOC(m1, m2, out, (left, right) -> {
			MatrixBlock lb = (MatrixBlock) left.getValue();
			MatrixBlock rb = (MatrixBlock) right.getValue();
			MatrixBlock combined = lb.binaryOperations(bop, rb);
			return new IndexedMatrixValue(
				new MatrixIndexes(left.getIndexes().getRowIndex(), left.getIndexes().getColumnIndex()), combined);
		}, key);

		return out;
	}

	private OOCStream<IndexedMatrixValue> elemDivOOC(OOCStream<IndexedMatrixValue> m1, OOCStream<IndexedMatrixValue> m2) {
		BinaryOperator div = InstructionUtils.parseBinaryOperator(Opcodes.DIV.toString());
		return elemOOC(m1, m2, div);
	}

	private OOCStream<IndexedMatrixValue> elemMultOOC(OOCStream<IndexedMatrixValue> m1, OOCStream<IndexedMatrixValue> m2) {
		BinaryOperator div = InstructionUtils.parseBinaryOperator(Opcodes.MULT.toString());
		return elemOOC(m1, m2, div);
	}

	private OOCStream<IndexedMatrixValue> elemMinusOOC(OOCStream<IndexedMatrixValue> m1, OOCStream<IndexedMatrixValue> m2) {
		BinaryOperator div = InstructionUtils.parseBinaryOperator(Opcodes.MINUS.toString());
		return elemOOC(m1, m2, div);
	}

	private OOCStream<IndexedMatrixValue> elemPlusOOC(OOCStream<IndexedMatrixValue> m1, double eps) {
		SubscribableTaskQueue<IndexedMatrixValue> out = new SubscribableTaskQueue<>();
		mapOOC(m1, out, blk -> {
			MatrixBlock res = ((MatrixBlock) blk.getValue())
				.scalarOperations(new RightScalarOperator(Plus.getPlusFnObject(), eps), null);
			return new IndexedMatrixValue(
				new MatrixIndexes(blk.getIndexes().getRowIndex(), blk.getIndexes().getColumnIndex()), res);
		});
		return out;
	}

	private OOCStream<IndexedMatrixValue> maskOOC(OOCStream<IndexedMatrixValue> mask, OOCStream<IndexedMatrixValue> m1) {
		SubscribableTaskQueue<IndexedMatrixValue> out = new SubscribableTaskQueue<>();
		Function<IndexedMatrixValue, MatrixIndexes> key = imv ->
			new MatrixIndexes(imv.getIndexes().getRowIndex(), imv.getIndexes().getColumnIndex());

		joinOOC(mask, m1, out, (left, right) -> {
			MatrixBlock lb = (MatrixBlock) left.getValue();
			MatrixBlock rb = (MatrixBlock) right.getValue();
			MatrixBlock combined = mask(lb, rb);
			return new IndexedMatrixValue(
				new MatrixIndexes(left.getIndexes().getRowIndex(), left.getIndexes().getColumnIndex()), combined);
		}, key);

		return out;
	}

	private MatrixBlock mask(MatrixBlock mask, MatrixBlock blk) {
		for(int i = 0; i < blk.getNumRows(); i++) {
			for(int j = 0; j < blk.getNumColumns(); j++) {
				if(mask.get(i,j) ==0) blk.set(i, j, 0);
			}
		}
		return blk;
	}
}
