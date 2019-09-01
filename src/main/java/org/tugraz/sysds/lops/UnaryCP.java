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

package org.tugraz.sysds.lops;

 
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;

public class UnaryCP extends Lop 
{
	public enum OperationTypes {
		NOT, ABS, SIN, COS, TAN, ASIN, ACOS, ATAN, SQRT, LOG, EXP, SINH, COSH, TANH,
		CAST_AS_SCALAR, CAST_AS_MATRIX, CAST_AS_FRAME, CAST_AS_DOUBLE, CAST_AS_INT, CAST_AS_BOOLEAN, 
		PRINT, ASSERT, NROW, NCOL, LENGTH, EXISTS, LINEAGE, ROUND, STOP, CEIL, FLOOR, CUMSUM, SOFTMAX
	}
	
	public static final String CAST_AS_SCALAR_OPCODE = "castdts";
	public static final String CAST_AS_MATRIX_OPCODE = "castdtm";
	public static final String CAST_AS_FRAME_OPCODE = "castdtf";
	public static final String CAST_AS_DOUBLE_OPCODE = "castvtd";
	public static final String CAST_AS_INT_OPCODE    = "castvti";
	public static final String CAST_AS_BOOLEAN_OPCODE = "castvtb";

	
	
	OperationTypes operation;

	/**
	 * Constructor to perform a scalar operation
	 * 
	 * @param input low-level operator 1
	 * @param op operation type
	 * @param dt data type
	 * @param vt value type
	 * @param et exec type
	 */
	public UnaryCP(Lop input, OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.UnaryCP, dt, vt);
		operation = op;
		this.addInput(input);
		input.addOutput(this);
		lps.setProperties(inputs, et);
	}
	
	public UnaryCP(Lop input, OperationTypes op, DataType dt, ValueType vt) {
		this(input, op, dt, vt, ExecType.CP);
	}

	@Override
	public String toString() {

		return "Operation: " + operation;

	}

	private String getOpCode() {
		switch (operation) {
		case NOT:
			return "!";

		case ABS:
			return "abs";

		case SIN:
			return "sin";

		case COS:
			return "cos";

		case TAN:
			return "tan";

		case ASIN:
			return "asin";

		case ACOS:
			return "acos";

		case ATAN:
			return "atan";

		case SINH:
			return "sinh";

		case COSH:
			return "cosh";

		case TANH:
			return "tanh";
			
		case SQRT:
			return "sqrt";

		case LOG:
			return "log";

		case ROUND:
			return "round";

		case EXP:
			return "exp";

		case PRINT:
			return "print";
		
		case ASSERT:
			return "assert";

		case CAST_AS_MATRIX:
			return CAST_AS_MATRIX_OPCODE;

		case CAST_AS_FRAME:
			return CAST_AS_FRAME_OPCODE;
			
		case STOP:
			return "stop";
			
		case CEIL:
			return "ceil";
			
		case FLOOR:
			return "floor";
		
		case CUMSUM:
			return "ucumk+";
			
		// CAST_AS_SCALAR, NROW, NCOL, LENGTH builtins take matrix as the input
		// and produces a scalar
		case CAST_AS_SCALAR:
			return CAST_AS_SCALAR_OPCODE; 

		case CAST_AS_DOUBLE:
			return CAST_AS_DOUBLE_OPCODE; 

		case CAST_AS_INT:
			return CAST_AS_INT_OPCODE; 

		case CAST_AS_BOOLEAN:
			return CAST_AS_BOOLEAN_OPCODE; 

		case NROW:   return "nrow";
		case NCOL:   return "ncol";
		case LENGTH: return "length";
		case EXISTS: return "exists";
		case LINEAGE: return "lineage";
		
		case SOFTMAX:
			return "softmax";
			
		default:
			throw new LopsException(this.printErrorLocation() + "Unknown operation: " + operation);
		}
	}
	
	@Override
	public String getInstructions(String input, String output) {
		return InstructionUtils.concatOperands(
			getExecType().name(),
			getOpCode(),
			getInputs().get(0).prepScalarInputOperand(getExecType()),
			prepOutputOperand(output));
	}
}
