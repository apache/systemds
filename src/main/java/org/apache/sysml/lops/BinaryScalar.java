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

package org.apache.sysml.lops;


 
import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Expression.*;

/**
 * Lop to perform binary scalar operations. Both inputs must be scalars.
 * Example i = j + k, i = i + 1. 
 */
public class BinaryScalar extends Lop 
{
	public enum OperationTypes {
		ADD, SUBTRACT, MULTIPLY, DIVIDE, MODULUS, INTDIV,
		LESS_THAN, LESS_THAN_OR_EQUALS, GREATER_THAN, GREATER_THAN_OR_EQUALS, EQUALS, NOT_EQUALS,
		AND, OR, XOR,
		LOG,POW,MAX,MIN,PRINT,IQSIZE,
		BW_AND, BW_OR, BW_XOR, BW_SHIFTL, BW_SHIFTR, //Bitwise operations
	}
	
	private final OperationTypes operation;
	
	/**
	 * Constructor to perform a scalar operation
	 * 
	 * @param input1 low-level operator 1
	 * @param input2 low-level operator 2
	 * @param op operation type
	 * @param dt data type
	 * @param vt value type
	 */
	public BinaryScalar(Lop input1, Lop input2, OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lop.Type.BinaryCP, dt, vt);		
		operation = op;		
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);

		boolean breaksAlignment = false; // this field does not carry any meaning for this lop
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.INVALID);
		lps.setProperties(inputs, ExecType.CP, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
	}

	@Override
	public String toString() {
		return "Operation: " + operation;
	}
	
	public OperationTypes getOperationType() {
		return operation;
	}

	@Override
	public String getInstructions(String input1, String input2, String output) throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		
		sb.append(getExecType());
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append( getOpcode(operation) );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(0).prepScalarInputOperand(getExecType()) );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(1).prepScalarInputOperand(getExecType()));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( prepOutputOperand(output));

		return sb.toString();
	}
	
	@Override
	public Lop.SimpleInstType getSimpleInstructionType() {
		return SimpleInstType.Scalar;
	}
	
	public static String getOpcode( OperationTypes op )
	{
		if( op == null )
			throw new UnsupportedOperationException("Unable to get opcode for 'null'.");
		
		switch ( op ) 
		{
			/* Arithmetic */
			case ADD:
				return "+";
			case SUBTRACT:
				return "-";
			case MULTIPLY:
				return "*";
			case DIVIDE:
				return "/";
			case MODULUS:
				return "%%";	
			case INTDIV:
				return "%/%";	
			case POW:	
				return "^";
				
			/* Relational */
			case LESS_THAN:
				return "<";
			case LESS_THAN_OR_EQUALS:
				return "<=";
			case GREATER_THAN:
				return ">";
			case GREATER_THAN_OR_EQUALS:
				return ">=";
			case EQUALS:
				return "==";
			case NOT_EQUALS:
				return "!=";
			
			/* Boolean */
			case AND:
				return "&&";
			case OR:
				return "||";

			/* Boolean built in binary function */
			case XOR:
				return "xor";
			case BW_AND:
				return "bitwAnd";
			case BW_OR:
				return "bitwOr";
			case BW_XOR:
				return "bitwXor";
			case BW_SHIFTL:
				return "bitwShiftL";
			case BW_SHIFTR:
				return "bitwShiftR";

			/* Builtin Functions */
			case LOG:
				return "log";
			case MIN:
				return "min"; 
			case MAX:
				return "max"; 
			
			case PRINT:
				return "print";
				
			case IQSIZE:
				return "iqsize"; 
				
			default:
				throw new UnsupportedOperationException("Instruction "
					+ "is not defined for BinaryScalar operator: " + op);
		}
	}
}
