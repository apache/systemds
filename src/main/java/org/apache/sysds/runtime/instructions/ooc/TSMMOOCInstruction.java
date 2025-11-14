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
import org.apache.sysds.lops.MMTSJ;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class TSMMOOCInstruction extends ComputationOOCInstruction {
	private final MMTSJType _type;

	protected TSMMOOCInstruction(OOCType type, Operator op, CPOperand in1, CPOperand out, MMTSJ.MMTSJType mmtsjType, String opcode, String istr) {
		super(type, op, in1, out, opcode, istr);
		_type = mmtsjType;
	}

	public static TSMMOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 3);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]); // the large matrix (streamed), columns <= blocksize
		CPOperand out = new CPOperand(parts[2]);
		MMTSJ.MMTSJType mmtsjType = MMTSJ.MMTSJType.valueOf(parts[3]);

		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator ba = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);

		return new TSMMOOCInstruction(OOCType.MMTSJ, ba, in1, out, mmtsjType, opcode, str);
	}

	@Override
	public void processInstruction( ExecutionContext ec ) {
		MatrixObject min = ec.getMatrixObject(input1);
		int nRows = (int) min.getDataCharacteristics().getRows();
		int nCols = (int) min.getDataCharacteristics().getCols();
		int bLen = min.getDataCharacteristics().getBlocksize();
		
		OOCStream<IndexedMatrixValue> qIn = min.getStreamHandle();
		BinaryOperator plus = InstructionUtils.parseBinaryOperator(Opcodes.PLUS.toString());

		//validation check TODO extend compiler to not create OOC otherwise
		if(    (_type.isLeft() && nCols > bLen)
			|| (_type.isRight() && nRows > bLen) )
		{
			throw new UnsupportedOperationException();
		}
		
		int dim = _type.isLeft() ? nCols : nRows;
		MatrixBlock resultBlock = null;

		OOCStream<MatrixBlock> tmpStream = createWritableStream();

		mapOOC(qIn, tmpStream,
			tmp -> ((MatrixBlock) tmp.getValue())
				.transposeSelfMatrixMultOperations(new MatrixBlock(), _type));

		MatrixBlock tmp;
		while ((tmp = tmpStream.dequeue()) != LocalTaskQueue.NO_MORE_TASKS) {
			if (resultBlock == null)
				resultBlock = tmp;
			else
				resultBlock.binaryOperationsInPlace(plus, tmp);
		}

		ec.setMatrixOutput(output.getName(), resultBlock);
	}
}
