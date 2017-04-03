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

import org.apache.sysml.lops.Binary.OperationTypes;
import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Expression.*;


/**
 * Lop to perform binary operation. Both inputs must be matrices or vectors. 
 * Example - A = B + C, where B and C are matrices or vectors.
 */

public class BinaryM extends Lop 
{
	
	public enum CacheType {
		RIGHT,
		RIGHT_PART,
	}
	
	public enum VectorType{
		COL_VECTOR,
		ROW_VECTOR,
	}
	
	private OperationTypes _operation;
	private CacheType _cacheType = null;
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
	 * @param partitioned true if partitioned
	 * @param colVector true if colVector
	 * @throws LopsException if LopsException occurs
	 */
	public BinaryM(Lop input1, Lop input2, OperationTypes op, DataType dt, ValueType vt, ExecType et, boolean partitioned, boolean colVector ) throws LopsException {
		super(Lop.Type.Binary, dt, vt);
		
		_operation = op;
		_cacheType = partitioned ? CacheType.RIGHT_PART : CacheType.RIGHT;
		_vectorType = colVector ? VectorType.COL_VECTOR : VectorType.ROW_VECTOR;
		
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		if(et == ExecType.MR) {
			lps.addCompatibility(JobType.GMR);
			lps.setProperties( inputs, ExecType.MR, ExecLocation.Map, breaksAlignment, aligner, definesMRJob );
		}
		else if(et == ExecType.SPARK) {
			lps.addCompatibility(JobType.INVALID);
			lps.setProperties( inputs, ExecType.SPARK, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
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

	public static boolean isOpcode(String opcode)
	{
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
	public String getInstructions(int input_index1, int input_index2, int output_index) throws LopsException {
		return getInstructions(
				String.valueOf(input_index1), 
				String.valueOf(input_index2), 
				String.valueOf(output_index));
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output) 
		throws LopsException 
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
		sb.append(_cacheType);
		
		sb.append( OPERAND_DELIMITOR );
		sb.append(_vectorType);
		
		return sb.toString();
	}
	
	@Override
	public boolean usesDistributedCache() {
		return true;
	}
	
	@Override
	public int[] distributedCacheInputIndex() {	
		// second input is from distributed cache
		return new int[]{2};
	}
}