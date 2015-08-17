/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.*;


/**
 * Lop to perform unary scalar operations. Example a = !b
 * 
 */

public class UnaryCP extends Lop 
{
	
	public enum OperationTypes {
		NOT, ABS, SIN, COS, TAN, ASIN, ACOS, ATAN, SQRT, LOG, EXP, CAST_AS_SCALAR, CAST_AS_MATRIX, CAST_AS_DOUBLE, CAST_AS_INT, CAST_AS_BOOLEAN, PRINT, NROW, NCOL, LENGTH, ROUND, STOP, CEIL, FLOOR, CUMSUM, NOTSUPPORTED
	};
	
	public static final String CAST_AS_SCALAR_OPCODE = "castdts";
	public static final String CAST_AS_MATRIX_OPCODE = "castdtm";
	public static final String CAST_AS_DOUBLE_OPCODE = "castvtd";
	public static final String CAST_AS_INT_OPCODE    = "castvti";
	public static final String CAST_AS_BOOLEAN_OPCODE = "castvtb";

	
	
	OperationTypes operation;

	/**
	 * Constructor to perform a scalar operation
	 * 
	 * @param input
	 * @param op
	 */

	public UnaryCP(Lop input, OperationTypes op, DataType dt, ValueType vt) {
		super(Lop.Type.UnaryCP, dt, vt);
		operation = op;
		this.addInput(input);
		input.addOutput(this);

		//This lop is executed in control program.
		boolean breaksAlignment = false; // this does not carry any information
											// for this lop
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.INVALID);
		this.lps.setProperties(inputs, ExecType.CP, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
	}

	@Override
	public String toString() {

		return "Operation: " + operation;

	}

	private String getOpCode() throws LopsException {
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

		case CAST_AS_MATRIX:
			return CAST_AS_MATRIX_OPCODE;
			
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

		case NROW:
			return "nrow";
		
		case NCOL:
			return "ncol";

		case LENGTH:
			return "length";

		default:
			throw new LopsException(this.printErrorLocation() + "Unknown operation: " + operation);
		}
	}
	
	@Override
	public String getInstructions(String input, String output)
			throws LopsException {
		StringBuilder sb = new StringBuilder();
		sb.append(getExecType());
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpCode() );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append(getInputs().get(0).prepScalarInputOperand(getExecType()));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();

	}
}
