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

 
import org.apache.sysds.lops.LopProperties.ExecType;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;


/**
 * Lop to perform following operations: with one operand -- NOT(A), ABS(A),
 * SQRT(A), LOG(A) with two operands where one of them is a scalar -- H=H*i,
 * H=H*5, EXP(A,2), LOG(A,2)
 * 
 */

public class Unary extends Lop 
{
	@SuppressWarnings("hiding")
	public enum OperationTypes {
		ADD, SUBTRACT, SUBTRACTRIGHT, MULTIPLY, MULTIPLY2, DIVIDE, MODULUS, INTDIV, MINUS1_MULTIPLY, 
		POW, POW2, LOG, MAX, MIN, NOT, ABS, SIN, COS, TAN, ASIN, ACOS, ATAN, SINH, COSH, TANH, SIGN, SQRT, EXP, Over, 
		LESS_THAN, LESS_THAN_OR_EQUALS, GREATER_THAN, GREATER_THAN_OR_EQUALS, EQUALS, NOT_EQUALS,
		AND, OR, XOR, BW_AND, BW_OR, BW_XOR, BW_SHIFTL, BW_SHIFTR,
		ROUND, CEIL, FLOOR, MR_IQM, INVERSE, CHOLESKY,
		CUMSUM, CUMPROD, CUMMIN, CUMMAX, CUMSUMPROD,
		ISNA, ISNAN, ISINF,
		SPROP, SIGMOID, SUBTRACT_NZ, LOG_NZ,
		CAST_AS_MATRIX, CAST_AS_FRAME,
		TYPEOF, DETECTSCHEMA,
		NOTSUPPORTED
	}

	private OperationTypes operation;
	private Lop valInput;
	
	//cp-specific parameters
	private int _numThreads = 1;
	private boolean _inplace = false;


	/**
	 * Constructor to perform a unary operation with 2 inputs
	 * 
	 * @param input1 low-level operator 1
	 * @param input2 low-level operator 2
	 * @param op operation type
	 * @param dt data type
	 * @param vt value type
	 * @param et execution type
	 */
	public Unary(Lop input1, Lop input2, OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.UNARY, dt, vt);
		init(input1, input2, op, dt, vt, et);
	}

	private void init(Lop input1, Lop input2, OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		operation = op;

		if (input1.getDataType() == DataType.MATRIX)
			valInput = input2;
		else
			valInput = input1;

		this.addInput(input1);
		input1.addOutput(this);
		this.addInput(input2);
		input2.addOutput(this);
		lps.setProperties(inputs, et);
	}

	/**
	 * Constructor to perform a unary operation with 1 input.
	 * 
	 * @param input1 low-level operator 1
	 * @param op operation type
	 * @param dt data type
	 * @param vt value type
	 * @param et execution type
	 * @param numThreads number of threads
	 * @param inplace inplace behavior
	 */
	public Unary(Lop input1, OperationTypes op, DataType dt, ValueType vt, ExecType et, int numThreads, boolean inplace) {
		super(Lop.Type.UNARY, dt, vt);
		init(input1, op, dt, vt, et);
		_numThreads = numThreads;
		_inplace = inplace;
	}

	private void init(Lop input1, OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		//sanity check
		if ( (op == OperationTypes.INVERSE || op == OperationTypes.CHOLESKY)
			 && et == ExecType.SPARK ) {
			throw new LopsException("Invalid exection type "+et.toString()+" for operation "+op.toString());
		}
		
		operation = op;
		valInput = null;

		this.addInput(input1);
		input1.addOutput(this);
		lps.setProperties(inputs, et);
	}

	@Override
	public String toString() {
		if (valInput != null)
			return "Operation: " + operation + " " + "Label: "
					+ valInput.getOutputParameters().getLabel()
					+ " input types " + this.getInputs().get(0).toString()
					+ " " + this.getInputs().get(1).toString();
		else
			return "Operation: " + operation + " " + "Label: N/A";
	}

	private String getOpcode() {
		return getOpcode(operation);
	}

	public static String getOpcode(OperationTypes op) {
		switch (op) {
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
		case SIGN:
			return "sign";
		case SQRT:
			return "sqrt";
		case EXP:
			return "exp";
		
		case LOG:
			return "log";
		
		case LOG_NZ:
			return "log_nz";
			
		case ROUND:
			return "round";

		case ADD:
			return "+";

		case SUBTRACT:
			return "-";

		case SUBTRACT_NZ:
			return "-nz";
		
		case SUBTRACTRIGHT:
			return "s-r";

		case MULTIPLY:
			return "*";

		case MULTIPLY2:
			return "*2";

		case MINUS1_MULTIPLY:
			return "1-*";
			
		case DIVIDE:
			return "/";

		case MODULUS:
			return "%%";
			
		case INTDIV:
			return "%/%";
			
		case Over:
			return "so";

		case POW:
			return "^";
		
		case POW2:
			return "^2";

		case GREATER_THAN:
			return ">";

		case GREATER_THAN_OR_EQUALS:
			return ">=";

		case LESS_THAN:
			return "<";

		case LESS_THAN_OR_EQUALS:
			return "<=";

		case EQUALS:
			return "==";

		case NOT_EQUALS:
			return "!=";

		case MAX:
			return "max";

		case MIN:
			return "min";
		
		case CEIL:
			return "ceil";
		
		case FLOOR:
			return "floor";
		
		case CUMSUM:
			return "ucumk+";
		
		case CUMPROD:
			return "ucum*";
		
		case CUMMIN:
			return "ucummin";
		
		case CUMMAX:
			return "ucummax";
		
		case CUMSUMPROD:
			return "ucumk+*";
		
		case INVERSE:
			return "inverse";
		
		case CHOLESKY:
			return "cholesky";
		
		case MR_IQM:
			return "qpick";

		case SPROP:
			return "sprop";
			
		case SIGMOID:
			return "sigmoid";

		case TYPEOF:
			return "typeOf";

		case DETECTSCHEMA:
			return "detectSchema";
		
		case CAST_AS_MATRIX:
			return UnaryCP.CAST_AS_MATRIX_OPCODE;

		case CAST_AS_FRAME:
			return UnaryCP.CAST_AS_FRAME_OPCODE;
		
		case AND: return "&&";
		case OR:  return "||";
		case XOR: return "xor";
		case BW_AND: return "bitwAnd";
		case BW_OR:  return "bitwOr";
		case BW_XOR: return "bitwXor";
		case BW_SHIFTL: return "bitwShiftL";
		case BW_SHIFTR: return "bitwShiftR";
		case ISNA: return "isna";
		case ISNAN: return "isnan";
		case ISINF: return "isinf";
		
		default:
			throw new LopsException(
				"Instruction not defined for Unary operation: " + op);
		}
	}
	
	public static boolean isMultiThreadedOp(OperationTypes op) {
		return op==OperationTypes.CUMSUM
			|| op==OperationTypes.CUMPROD
			|| op==OperationTypes.CUMMIN
			|| op==OperationTypes.CUMMAX
			|| op==OperationTypes.CUMSUMPROD
			|| op==OperationTypes.EXP
			|| op==OperationTypes.LOG
			|| op==OperationTypes.SIGMOID;
	}
	
	@Override
	public String getInstructions(String input1, String output) {
		//sanity check number of operands
		if( getInputs().size() != 1 ) {
			throw new LopsException(printErrorLocation() + "Invalid number of operands ("
					+ getInputs().size() + ") for an Unary opration: " + operation);		
		}
		
		// Unary operators with one input
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input1) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output) );
		
		//num threads for cumulative cp ops
		if( getExecType() == ExecType.CP && isMultiThreadedOp(operation) ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _numThreads );
			sb.append( OPERAND_DELIMITOR );
			sb.append( _inplace );
		}
		
		return sb.toString();
	}

	@Override
	public String getInstructions(String input1, String input2, String output) {
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		
		sb.append( OPERAND_DELIMITOR );
		if ( getInputs().get(0).getDataType() == DataType.SCALAR )
			sb.append( getInputs().get(0).prepScalarInputOperand(getExecType()));
		else
			sb.append( getInputs().get(0).prepInputOperand(input1));
		
		sb.append( OPERAND_DELIMITOR );
		if ( getInputs().get(1).getDataType() == DataType.SCALAR )
			sb.append( getInputs().get(1).prepScalarInputOperand(getExecType()));
		else 
			sb.append( getInputs().get(1).prepInputOperand(input2));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();
	}
}
