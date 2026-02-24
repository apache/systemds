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

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.functionobjects.OffsetColumnIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class AppendOOCInstruction extends BinaryOOCInstruction {

	public enum AppendType {
		CBIND
	}

	protected final AppendType _type;

	protected AppendOOCInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, AppendType type,
		String opcode, String istr) {
		super(OOCType.Append, op, in1, in2, out, opcode, istr);
		_type = type;
	}

	public static AppendOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 5, 4);

		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[parts.length-2]);
		boolean cbind = Boolean.parseBoolean(parts[parts.length-1]);

		if(in1.getDataType() != Types.DataType.MATRIX || in2.getDataType() != Types.DataType.MATRIX || !cbind){
			throw new DMLRuntimeException("Only matrix-matrix cbind is supported");
		}
		AppendType type = AppendType.CBIND;

		Operator op = new ReorgOperator(OffsetColumnIndex.getOffsetColumnIndexFnObject(-1));
		return new AppendOOCInstruction(op, in1, in2, out, type, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixObject in1 = ec.getMatrixObject(input1);
		MatrixObject in2 = ec.getMatrixObject(input2);
		validateInput(in1, in2);

		OOCStream<IndexedMatrixValue> qIn1 = in1.getStreamHandle();
		OOCStream<IndexedMatrixValue> qIn2 = in2.getStreamHandle();

		int blksize = in1.getBlocksize();
		int rem1 = (int) in1.getNumColumns()%blksize;
		int rem2 = (int) in2.getNumColumns()%blksize;
		int cblk1 = (int) in1.getDataCharacteristics().getNumColBlocks();
		int cblk2 = (int) in2.getDataCharacteristics().getNumColBlocks();

		if(rem1+rem2 == 0){
			// no shifting needed
			OOCStream<IndexedMatrixValue> out = new SubscribableTaskQueue<>();
			mapOOC(qIn2, out, imv -> new IndexedMatrixValue(
				new MatrixIndexes(imv.getIndexes().getRowIndex(), cblk1+imv.getIndexes().getColumnIndex()), imv.getValue()));

			ec.getMatrixObject(output).setStreamHandle(mergeOOCStreams(List.of(qIn1, out)));
			return;
		}

		List<OOCStream<IndexedMatrixValue>> split1 = splitOOCStream(qIn1, imv -> imv.getIndexes().getColumnIndex()==cblk1? 1 : 0, 2);
		List<OOCStream<IndexedMatrixValue>> split2 = splitOOCStream(qIn2, imv -> (int) imv.getIndexes().getColumnIndex()-1, cblk2);

		OOCStream<IndexedMatrixValue> head = split1.get(0);
		OOCStream<IndexedMatrixValue> lastCol = split1.get(1);
		OOCStream<IndexedMatrixValue> firstCol = split2.get(0);

		CachingStream firstColCache = new CachingStream(firstCol);
		OOCStream<IndexedMatrixValue> firstColForCritical = firstColCache.getReadStream();
		OOCStream<IndexedMatrixValue> firstColForTail = firstColCache.getReadStream();

		SubscribableTaskQueue<IndexedMatrixValue> out = new SubscribableTaskQueue<>();
		Function<IndexedMatrixValue, MatrixIndexes> rowKey = imv -> new MatrixIndexes(imv.getIndexes().getRowIndex(), 1);

		// combine cols both matrices
		joinOOC(lastCol, firstColForCritical, out, (left, right) -> {
			MatrixBlock lb = (MatrixBlock) left.getValue();
			MatrixBlock rb = (MatrixBlock) right.getValue();
			int stop = cblk2>1? blksize-rem1 : rem2;
			MatrixBlock combined = cbindBlocks(lb, sliceCols(rb, 0, stop));
			return new IndexedMatrixValue(
				new MatrixIndexes(left.getIndexes().getRowIndex(), left.getIndexes().getColumnIndex()), combined);
		}, rowKey);

		List<OOCStream<IndexedMatrixValue>> outStreams = new ArrayList<>();
		outStreams.add(head);
		outStreams.add(out);

		// shift cols second matrix
		OOCStream<IndexedMatrixValue> fst = firstColForTail;
		OOCStream<IndexedMatrixValue> sec = null;
		for(int i=0; i<cblk2-1; i++){
			out = new SubscribableTaskQueue<>();
			CachingStream secCachingStream = new CachingStream(split2.get(i+1));
			sec = secCachingStream.getReadStream();

			int finalI = i;
			joinOOC(fst, sec, out, (left, right) -> {
				MatrixBlock lb = (MatrixBlock) left.getValue();
				MatrixBlock rb = (MatrixBlock) right.getValue();
				int stop = finalI+2==cblk2 ? rem2 : blksize-rem1;
				MatrixBlock combined = cbindBlocks(sliceCols(lb, blksize-rem1, blksize), sliceCols(rb, 0, stop));
				return new IndexedMatrixValue(
					new MatrixIndexes(left.getIndexes().getRowIndex(), cblk1 + left.getIndexes().getColumnIndex()),
					combined);
			}, rowKey);

			fst = secCachingStream.getReadStream();
			outStreams.add(out);
		}

		if(rem1+rem2 > blksize){
			// overflow
			int remSize = (rem1+rem2)%blksize;
			out = new SubscribableTaskQueue<>();
			mapOOC(fst, out, imv -> new IndexedMatrixValue(
				new MatrixIndexes(imv.getIndexes().getRowIndex(), cblk1+imv.getIndexes().getColumnIndex()), 
				sliceCols((MatrixBlock) imv.getValue(), rem2-remSize, rem2)));

			outStreams.add(out);
		}
		ec.getMatrixObject(output).setStreamHandle(mergeOOCStreams(outStreams));
	}

	public AppendType getAppendType() {
		return _type;
	}

	private void validateInput(MatrixObject m1, MatrixObject m2) {
		if(_type == AppendType.CBIND && m1.getNumRows() != m2.getNumRows()) {
			throw new DMLRuntimeException(
				"Append-cbind is not possible for input matrices " + input1.getName() + " and " + input2.getName()
					+ " with different number of rows: " + m1.getNumRows() + " vs " + m2.getNumRows());
		}
	}

	private static MatrixBlock sliceCols(MatrixBlock in, int colStart, int colEndExclusive) {
		// slice is inclusive
		return in.slice(0, in.getNumRows()-1, colStart, colEndExclusive-1);
	}

	private static MatrixBlock cbindBlocks(MatrixBlock left, MatrixBlock right) {
		return left.append(right, new MatrixBlock());
	}
}
