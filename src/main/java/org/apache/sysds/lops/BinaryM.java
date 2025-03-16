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

package org.apache.sysds.lops;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.ValueType;


/**
 * Lop to perform binary operation. Both inputs must be matrices or vectors. 
 * Example - A = B + C, where B and C are matrices or vectors.
 */

public class BinaryM extends Lop 
{
	public enum VectorType{
		COL_VECTOR,
		ROW_VECTOR,
	}
	
	private OpOp2 _operation;
	private VectorType _vectorType = null; 
	
	/**
	 * Constructor to perform a binary operation.
	 * 
	 * @param input1 low-level operator 1
	 * @param input2 low-level operator 2
	 * @param op operation type
	 * @param dt data type
	 * @param vt value type
	 * @param et exec type
	 * @param colVector true if colVector
	 */
	public BinaryM(Lop input1, Lop input2, OpOp2 op, DataType dt, ValueType vt, ExecType et, boolean colVector ) {
		super(Lop.Type.Binary, dt, vt);
		
		_operation = op;
		_vectorType = colVector ? VectorType.COL_VECTOR : VectorType.ROW_VECTOR;
		
		addInput(input1);
		addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		
		if(et == ExecType.SPARK) {
			lps.setProperties( inputs, ExecType.SPARK);
		}
		else {
			throw new LopsException("Incorrect execution type for BinaryM lop:" + et.name());
		}
	}
	
	@Override
	public String toString() {
		return " Operation: " + _operation;
	}

	@Override
	public Lop getBroadcastInput() {
		if (getExecType() != ExecType.SPARK)
			return null;
		return inputs.get(1);
	}

	public OpOp2 getOperationType() {
		return _operation;
	}

	private String getOpcode() {
		return getOpcode(_operation);
	}
	
	public static String getOpcode(OpOp2 op) {
		return "map"+op.toString();
	}

	public static boolean isOpcode(String opcode) {
		return opcode.equals(Opcodes.MAPPLUS.toString()) || opcode.equals(Opcodes.MAPMINUS.toString()) ||
			opcode.equals(Opcodes.MAPMULT.toString()) || opcode.equals(Opcodes.MAPDIV.toString()) ||
			opcode.equals(Opcodes.MAPMOD.toString()) || opcode.equals(Opcodes.MAPINTDIV.toString()) ||
			opcode.equals(Opcodes.MAPLT.toString()) || opcode.equals(Opcodes.MAPLE.toString()) ||
			opcode.equals(Opcodes.MAPGT.toString()) || opcode.equals(Opcodes.MAPGE.toString()) ||
			opcode.equals(Opcodes.MAPEQ.toString()) || opcode.equals(Opcodes.MAPNEQ.toString()) ||
			opcode.equals(Opcodes.MAPAND.toString()) || opcode.equals(Opcodes.MAPOR.toString()) ||
			opcode.equals(Opcodes.MAPMIN.toString()) || opcode.equals(Opcodes.MAPMAX.toString()) ||
			opcode.equals(Opcodes.MAPPOW.toString()) || opcode.equals(Opcodes.MAPMINUS1_MULT.toString());
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output) {
		return InstructionUtils.concatOperands(
			getExecType().name(),
			getOpcode(),
			getInputs().get(0).prepInputOperand(input1),
			getInputs().get(1).prepInputOperand(input2),
			prepOutputOperand(output),
			"RIGHT",
			_vectorType.name());
	}
}
