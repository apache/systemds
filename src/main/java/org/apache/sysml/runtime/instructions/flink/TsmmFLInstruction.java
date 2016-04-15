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

package org.apache.sysml.runtime.instructions.flink;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.sysml.lops.MMTSJ.MMTSJType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.FlinkExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.flink.utils.DataSetAggregateUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.operators.Operator;

public class TsmmFLInstruction extends UnaryFLInstruction {

	private MMTSJType _type = null;
	private FLInstruction.FLINSTRUCTION_TYPE _fltype = null;

	public TsmmFLInstruction(Operator op, CPOperand in1, CPOperand out, MMTSJType type, String opcode, String istr) {
		super(op, in1, out, opcode, istr);
		_fltype = FLInstruction.FLINSTRUCTION_TYPE.TSMM;
		_type = MMTSJType.LEFT;
	}

	public static TsmmFLInstruction parseInstruction(String str) throws DMLRuntimeException {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		//check supported opcode
		if (!opcode.equalsIgnoreCase("tsmm")) {
			throw new DMLRuntimeException("TsmmFLInstruction.parseInstruction():: Unknown opcode " + opcode);
		}

		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		MMTSJType type = MMTSJType.valueOf(parts[3]);

		return new TsmmFLInstruction(null, in1, out, type, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException {
		FlinkExecutionContext fec = (FlinkExecutionContext) ec;

		//get input
		DataSet<Tuple2<MatrixIndexes, MatrixBlock>> in = fec.getBinaryBlockDataSetHandleForVariable(input1.getName());
		DataSet<MatrixBlock> tmp = in.map(new DataSetTSMMFunction(_type));
		MatrixBlock out = DataSetAggregateUtils.sumStable(tmp);

		//put output block into symbol table (no lineage because single block)
		//this also includes implicit maintenance of matrix characteristics
		fec.setMatrixOutput(output.getName(), out);
	}

	private static class DataSetTSMMFunction implements MapFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixBlock> {

		private static final long serialVersionUID = 2935770425858019666L;

		private MMTSJType _type = null;

		public DataSetTSMMFunction(MMTSJType type) {
			_type = type;
		}

		@Override
		public MatrixBlock map(Tuple2<MatrixIndexes, MatrixBlock> value) throws Exception {
			return value.f1.transposeSelfMatrixMultOperations(new MatrixBlock(), _type);
		}
	}
}
