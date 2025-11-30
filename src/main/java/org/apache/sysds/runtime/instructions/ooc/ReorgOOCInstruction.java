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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.functionobjects.SortIndex;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.util.DataConverter;

public class ReorgOOCInstruction extends ComputationOOCInstruction {
	// sort-specific attributes (to enable variable attributes)
	private final CPOperand _col;
	private final CPOperand _desc;
	private final CPOperand _ixret;

	protected ReorgOOCInstruction(ReorgOperator op, CPOperand in1, CPOperand out, String opcode, String istr) {
		this(op, in1, out, null, null, null, opcode, istr);
	}

	private ReorgOOCInstruction(Operator op, CPOperand in, CPOperand out, CPOperand col, CPOperand desc, CPOperand ixret,
		String opcode, String istr) {
		super(OOCType.Reorg, op, in, out, opcode, istr);
		_col = col;
		_desc = desc;
		_ixret = ixret;
	}

	public static ReorgOOCInstruction parseInstruction(String str) {
		CPOperand in = new CPOperand("", Types.ValueType.UNKNOWN, Types.DataType.UNKNOWN);
		CPOperand out = new CPOperand("", Types.ValueType.UNKNOWN, Types.DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if (opcode.equalsIgnoreCase(Opcodes.TRANSPOSE.toString())) {
			InstructionUtils.checkNumFields(str, 2, 3);
			in.split(parts[1]);
			out.split(parts[2]);

			ReorgOperator reorg = new ReorgOperator(SwapIndex.getSwapIndexFnObject());
			return new ReorgOOCInstruction(reorg, in, out, opcode, str);
		}
		else if (opcode.equalsIgnoreCase(Opcodes.SORT.toString())) {
			InstructionUtils.checkNumFields(str, 5,6);
			in.split(parts[1]);
			out.split(parts[5]);
			CPOperand col = new CPOperand(parts[2]);
			CPOperand desc = new CPOperand(parts[3]);
			CPOperand ixret = new CPOperand(parts[4]);
			int k = Integer.parseInt(parts[6]);
			return new ReorgOOCInstruction(new ReorgOperator(new SortIndex(1,false,false), k),
				in, out, col, desc, ixret, opcode, str);
		}
		else
			throw new NotImplementedException();
	}

	public void processInstruction( ExecutionContext ec ) {
		// Create thread and process the transpose operation
		MatrixObject min = ec.getMatrixObject(input1);


		ReorgOperator r_op = (ReorgOperator) _optr;

		if(r_op.fn instanceof SortIndex) {
			//additional attributes for sort
			int[] cols = _col.getDataType().isMatrix() ? DataConverter.convertToIntVector(ec.getMatrixInput(_col.getName())) :
				new int[]{(int)ec.getScalarInput(_col).getLongValue()};
			boolean desc = ec.getScalarInput(_desc).getBooleanValue();
			boolean ixret = ec.getScalarInput(_ixret).getBooleanValue();
			r_op = r_op.setFn(new SortIndex(cols, desc, ixret));

			// For now, we reuse the CP instruction
			// In future, we could optimize by building the permutation and streaming blocks column by column
			MatrixBlock matBlock = min.acquireRead();
			MatrixBlock soresBlock = matBlock.reorgOperations(r_op, new MatrixBlock(), 0, 0, 0);
			if (_col.getDataType().isMatrix())
				ec.releaseMatrixInput(_col.getName());
			ec.releaseMatrixInput(input1.getName());
			ec.setMatrixOutput(output.getName(), soresBlock);
		} else if (r_op.fn instanceof SwapIndex) {
			OOCStream<IndexedMatrixValue> qIn = min.getStreamHandle();
			OOCStream<IndexedMatrixValue> qOut = createWritableStream();
			ec.getMatrixObject(output).setStreamHandle(qOut);
			// Transpose operation
			mapOOC(qIn, qOut, tmp -> {
				MatrixBlock inBlock = (MatrixBlock) tmp.getValue();
				long oldRowIdx = tmp.getIndexes().getRowIndex();
				long oldColIdx = tmp.getIndexes().getColumnIndex();

				MatrixBlock outBlock = inBlock.reorgOperations((ReorgOperator) _optr, new MatrixBlock(), -1, -1, -1);
				return new IndexedMatrixValue(new MatrixIndexes(oldColIdx, oldRowIdx), outBlock);
			});
		}
	}
}
