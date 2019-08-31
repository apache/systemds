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

import org.tugraz.sysds.lops.Binary.OperationTypes;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;


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
	
	private OperationTypes _operation;
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
	public BinaryM(Lop input1, Lop input2, OperationTypes op, DataType dt, ValueType vt, ExecType et, boolean colVector ) {
		super(Lop.Type.Binary, dt, vt);
		
		_operation = op;
		_vectorType = colVector ? VectorType.COL_VECTOR : VectorType.ROW_VECTOR;
		
		this.addInput(input1);
		this.addInput(input2);
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
	public String toString() 
	{
		return " Operation: " + _operation;
	}

	/**
	 * method to get operation type
	 * @return operation type
	 */
	 
	public OperationTypes getOperationType()
	{
		return _operation;
	}

	private String getOpcode()
	{
		return getOpcode( _operation );
	}
	
	public static String getOpcode( OperationTypes op ) {
		switch(op) {
		/* Arithmetic */
		case ADD:
			return "map+";
		case SUBTRACT:
			return "map-";
		case MULTIPLY:
			return "map*";
		case DIVIDE:
			return "map/";
		case MODULUS:
			return "map%%";	
		case INTDIV:
			return "map%/%";
		case MINUS1_MULTIPLY:
			return "map1-*";	
		
		/* Relational */
		case LESS_THAN:
			return "map<";
		case LESS_THAN_OR_EQUALS:
			return "map<=";
		case GREATER_THAN:
			return "map>";
		case GREATER_THAN_OR_EQUALS:
			return "map>=";
		case EQUALS:
			return "map==";
		case NOT_EQUALS:
			return "map!=";
		
			/* Boolean */
		case AND:
			return "map&&";
		case OR:
			return "map||";
		
		
		/* Builtin Functions */
		case MIN:
			return "mapmin";
		case MAX:
			return "mapmax";
		case POW:
			return "map^";
			
		default:
			throw new UnsupportedOperationException("Instruction is not defined for Binary operation: " + op);
		}
	}

	public static boolean isOpcode(String opcode) {
		return opcode.equals("map+") || opcode.equals("map-") ||
			   opcode.equals("map*") || opcode.equals("map/") ||
			   opcode.equals("map%%") || opcode.equals("map%/%") ||
			   opcode.equals("map<") || opcode.equals("map<=") ||
			   opcode.equals("map>") || opcode.equals("map>=") ||
			   opcode.equals("map==") || opcode.equals("map!=") ||
			   opcode.equals("map&&") || opcode.equals("map||") ||
			   opcode.equals("mapmin") || opcode.equals("mapmax") ||
			   opcode.equals("map^") || opcode.equals("map1-*");
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output) 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() ); 
		
		sb.append( OPERAND_DELIMITOR );
		sb.append ( getInputs().get(0).prepInputOperand(input1));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append ( getInputs().get(1).prepInputOperand(input2));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append("RIGHT");
		
		sb.append( OPERAND_DELIMITOR );
		sb.append(_vectorType);
		
		return sb.toString();
	}
}
