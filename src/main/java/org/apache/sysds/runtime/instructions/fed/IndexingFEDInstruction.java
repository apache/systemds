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

package org.apache.sysds.runtime.instructions.fed;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LeftIndex;
import org.apache.sysds.lops.RightIndex;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.util.IndexRange;

public abstract class IndexingFEDInstruction extends UnaryFEDInstruction {
	protected final CPOperand rowLower, rowUpper, colLower, colUpper;

	protected IndexingFEDInstruction(CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu,
		CPOperand out, String opcode, String istr) {
		super(FEDInstruction.FEDType.MatrixIndexing, null, in, out, opcode, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
	}

	protected IndexingFEDInstruction(CPOperand lhsInput, CPOperand rhsInput, CPOperand rl, CPOperand ru, CPOperand cl,
		CPOperand cu, CPOperand out, String opcode, String istr) {
		super(FEDInstruction.FEDType.MatrixIndexing, null, lhsInput, rhsInput, out, opcode, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
	}

	protected IndexRange getIndexRange(ExecutionContext ec) {
		return new IndexRange( // rl, ru, cl, ru
			(int) (ec.getScalarInput(rowLower).getLongValue() - 1),
			(int) (ec.getScalarInput(rowUpper).getLongValue() - 1),
			(int) (ec.getScalarInput(colLower).getLongValue() - 1),
			(int) (ec.getScalarInput(colUpper).getLongValue() - 1));
	}

	public static IndexingFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if(opcode.equalsIgnoreCase(RightIndex.OPCODE)) {
			if(parts.length == 7) {
				CPOperand in, rl, ru, cl, cu, out;
				in = new CPOperand(parts[1]);
				rl = new CPOperand(parts[2]);
				ru = new CPOperand(parts[3]);
				cl = new CPOperand(parts[4]);
				cu = new CPOperand(parts[5]);
				out = new CPOperand(parts[6]);
				if(in.getDataType() == Types.DataType.MATRIX)
					return new MatrixIndexingFEDInstruction(in, rl, ru, cl, cu, out, opcode, str);
				else
					throw new DMLRuntimeException("Can index only on matrices, frames, and lists in federated.");
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		}
		else if(opcode.equalsIgnoreCase(LeftIndex.OPCODE)) {
			throw new DMLRuntimeException("Left indexing not implemented for federated operations.");
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a MatrixIndexingFEDInstruction: " + str);
		}
	}
}
