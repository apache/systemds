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

import java.util.concurrent.CompletableFuture;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.lops.MapMultChain.ChainType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
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
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;

public class MapMMChainOOCInstruction extends ComputationOOCInstruction {
	private final ChainType _type;

	protected MapMMChainOOCInstruction(OOCType type, Operator op, CPOperand in1, CPOperand in2, CPOperand in3,
		CPOperand out, ChainType chainType, String opcode, String istr) {
		super(type, op, in1, in2, in3, out, opcode, istr);
		_type = chainType;
	}

	public static MapMMChainOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 4, 5);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);

		if (parts.length == 5) {
			CPOperand out = new CPOperand(parts[3]);
			ChainType type = ChainType.valueOf(parts[4]);
			return new MapMMChainOOCInstruction(OOCType.MAPMMCHAIN, null, in1, in2, null, out, type, opcode, str);
		}
		else { //parts.length==6
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[4]);
			ChainType type = ChainType.valueOf(parts[5]);
			return new MapMMChainOOCInstruction(OOCType.MAPMMCHAIN, null, in1, in2, in3, out, type, opcode, str);
		}
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixObject min = ec.getMatrixObject(input1);
		MatrixObject mv = ec.getMatrixObject(input2);
		OOCStream<IndexedMatrixValue> qV = mv.getStreamHandle();

		OOCStream<IndexedMatrixValue> qOut = createWritableStream();
		addOutStream(qOut);
		ec.getMatrixObject(output).setStreamHandle(qOut);

		OOCStream<IndexedMatrixValue> qInX = min.getStreamHandle();
		boolean createdCache = !qInX.hasStreamCache();
		CachingStream xCache = createdCache ? new CachingStream(qInX) : qInX.getStreamCache();

		long numRowBlocksL = min.getDataCharacteristics().getNumRowBlocks();
		long numColBlocksL = min.getDataCharacteristics().getNumColBlocks();
		int numRowBlocks = Math.toIntExact(numRowBlocksL);
		int numColBlocks = Math.toIntExact(numColBlocksL);
		long vRows = mv.getDataCharacteristics().getRows();

		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator mmOp = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
		BinaryOperator plus = InstructionUtils.parseBinaryOperator(Opcodes.PLUS.toString());

		boolean hasV = !mv.getDataCharacteristics().rowsKnown() || vRows > 0;
		if(!hasV && _type != ChainType.XtXvy)
			throw new DMLRuntimeException("MMChain requires non-empty v for chain type " + _type);

		OOCStream<IndexedMatrixValue> qU;
		CompletableFuture<Void> uFuture;

		if(!hasV && _type == ChainType.XtXvy) {
			MatrixObject mw = ec.getMatrixObject(input3);
			OOCStream<IndexedMatrixValue> qW = mw.getStreamHandle();
			OOCStream<IndexedMatrixValue> qNegW = createWritableStream();
			RightScalarOperator negOp = new RightScalarOperator(Multiply.getMultiplyFnObject(), -1);

			uFuture = mapOOC(qW, qNegW, tmp -> {
				MatrixBlock wBlock = (MatrixBlock) tmp.getValue();
				MatrixBlock neg = wBlock.scalarOperations(negOp, new MatrixBlock());
				return new IndexedMatrixValue(new MatrixIndexes(tmp.getIndexes().getRowIndex(), 1L), neg);
			});
			qU = qNegW;
		}
		else {
			OOCStream<IndexedMatrixValue> qPartialXv = createWritableStream();
			OOCStream<IndexedMatrixValue> qXv = createWritableStream();
			OOCStream<IndexedMatrixValue> qInXv = xCache.getReadStream();

			CompletableFuture<Void> mapXvFuture = broadcastJoinOOC(qInXv, qV, qPartialXv, (x, v) -> {
				MatrixBlock xBlock = (MatrixBlock) x.getValue();
				MatrixBlock vBlock = (MatrixBlock) v.getValue().getValue();
				MatrixBlock partial = xBlock.aggregateBinaryOperations(xBlock, vBlock, new MatrixBlock(), mmOp);
				return new IndexedMatrixValue(new MatrixIndexes(x.getIndexes().getRowIndex(), 1L), partial);
			}, tmp -> tmp.getIndexes().getColumnIndex(), tmp -> tmp.getIndexes().getRowIndex());

			CompletableFuture<Void> reduceXvFuture = groupedReduceOOC(qPartialXv, qXv, (left, right) -> {
				MatrixBlock mb = ((MatrixBlock) left.getValue()).binaryOperationsInPlace(plus, right.getValue());
				left.setValue(mb);
				return left;
			}, numColBlocks);

			if(_type.isWeighted()) {
				MatrixObject mw = ec.getMatrixObject(input3);
				OOCStream<IndexedMatrixValue> qW = mw.getStreamHandle();
				OOCStream<IndexedMatrixValue> qWeighted = createWritableStream();
				BinaryOperator weightOp = InstructionUtils.parseBinaryOperator(
					_type == ChainType.XtwXv ? Opcodes.MULT.toString() : Opcodes.MINUS.toString());

				uFuture = broadcastJoinOOC(qXv, qW, qWeighted, (u, w) -> {
					MatrixBlock uBlock = (MatrixBlock) u.getValue();
					MatrixBlock wBlock = (MatrixBlock) w.getValue().getValue();
					MatrixBlock updated = uBlock.binaryOperationsInPlace(weightOp, wBlock);
					u.setValue(updated);
					return u;
				}, tmp -> tmp.getIndexes().getRowIndex(), tmp -> tmp.getIndexes().getRowIndex());
				qU = qWeighted;
			}
			else {
				uFuture = reduceXvFuture;
				qU = qXv;
			}

			mapXvFuture.exceptionally(err -> {
				qOut.propagateFailure(DMLRuntimeException.of(err));
				return null;
			});
		}

		OOCStream<IndexedMatrixValue> qInXt = xCache.getReadStream();
		OOCStream<IndexedMatrixValue> qPartialXt = createWritableStream();
		CompletableFuture<Void> joinXtFuture = broadcastJoinOOC(qInXt, qU, qPartialXt, (x, u) -> {
			MatrixBlock xBlock = (MatrixBlock) x.getValue();
			MatrixBlock uBlock = (MatrixBlock) u.getValue().getValue();
			MatrixBlock partial = multTransposeVector(xBlock, uBlock);
			return new IndexedMatrixValue(new MatrixIndexes(x.getIndexes().getColumnIndex(), 1L), partial);
		}, tmp -> tmp.getIndexes().getRowIndex(), tmp -> tmp.getIndexes().getRowIndex());

		CompletableFuture<Void> outFuture = groupedReduceOOC(qPartialXt, qOut, (left, right) -> {
			MatrixBlock mb = ((MatrixBlock) left.getValue()).binaryOperationsInPlace(plus, right.getValue());
			left.setValue(mb);
			return left;
		}, numRowBlocks);

		outFuture.whenComplete((res, err) -> {
			if(createdCache)
				xCache.scheduleDeletion();
		});

		uFuture.exceptionally(err -> {
			qOut.propagateFailure(DMLRuntimeException.of(err));
			return null;
		});
		joinXtFuture.exceptionally(err -> {
			qOut.propagateFailure(DMLRuntimeException.of(err));
			return null;
		});
		outFuture.exceptionally(err -> {
			qOut.propagateFailure(DMLRuntimeException.of(err));
			return null;
		});
	}

	private static MatrixBlock multTransposeVector(MatrixBlock x, MatrixBlock u) {
		int rows = x.getNumRows();
		int cols = x.getNumColumns();
		MatrixBlock out = new MatrixBlock(cols, 1, false);
		out.allocateDenseBlock();
		double[] outVals = out.getDenseBlockValues();

		if(x.isInSparseFormat()) {
			SparseBlock a = x.getSparseBlock();
			if(a != null) {
				if(u.isInSparseFormat()) {
					for(int i = 0; i < rows; i++) {
						if(a.isEmpty(i))
							continue;
						double uval = u.get(i, 0);
						if(uval == 0)
							continue;
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						double[] avals = a.values(i);
						for(int k = apos; k < apos + alen; k++)
							outVals[aix[k]] += uval * avals[k];
					}
				}
				else {
					double[] uvals = u.getDenseBlockValues();
					for(int i = 0; i < rows; i++) {
						if(a.isEmpty(i))
							continue;
						double uval = uvals[i];
						if(uval == 0)
							continue;
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						double[] avals = a.values(i);
						for(int k = apos; k < apos + alen; k++)
							outVals[aix[k]] += uval * avals[k];
					}
				}
			}
		}
		else {
			DenseBlock a = x.getDenseBlock();
			if(u.isInSparseFormat()) {
				for(int i = 0; i < rows; i++) {
					double uval = u.get(i, 0);
					if(uval == 0)
						continue;
					double[] avals = a.values(i);
					int apos = a.pos(i);
					for(int j = 0; j < cols; j++)
						outVals[j] += uval * avals[apos + j];
				}
			}
			else {
				double[] uvals = u.getDenseBlockValues();
				for(int i = 0; i < rows; i++) {
					double uval = uvals[i];
					if(uval == 0)
						continue;
					double[] avals = a.values(i);
					int apos = a.pos(i);
					for(int j = 0; j < cols; j++)
						outVals[j] += uval * avals[apos + j];
				}
			}
		}

		out.recomputeNonZeros();
		out.examSparsity();
		return out;
	}
}
