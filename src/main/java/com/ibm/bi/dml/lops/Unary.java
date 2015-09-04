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
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


/**
 * Lop to perform following operations: with one operand -- NOT(A), ABS(A),
 * SQRT(A), LOG(A) with two operands where one of them is a scalar -- H=H*i,
 * H=H*5, EXP(A,2), LOG(A,2)
 * 
 */

public class Unary extends Lop 
{
	
	public enum OperationTypes {
		ADD, SUBTRACT, SUBTRACTRIGHT, MULTIPLY, MULTIPLY2, DIVIDE, MODULUS, INTDIV, POW, POW2, LOG, MAX, MIN, NOT, ABS, SIN, COS, TAN, ASIN, ACOS, ATAN, SQRT, EXP, Over, LESS_THAN, LESS_THAN_OR_EQUALS, GREATER_THAN, GREATER_THAN_OR_EQUALS, EQUALS, NOT_EQUALS, ROUND, CEIL, FLOOR, MR_IQM, INVERSE,
		CUMSUM, CUMPROD, CUMMIN, CUMMAX,
		SPROP, SIGMOID, SELP, SUBTRACT_NZ, LOG_NZ,
		NOTSUPPORTED
	};

	OperationTypes operation;

	Lop valInput;

	/**
	 * Constructor to perform a unary operation with 2 inputs
	 * 
	 * @param input
	 * @param op
	 */

	public Unary(Lop input1, Lop input2, OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.UNARY, dt, vt);
		init(input1, input2, op, dt, vt, et);
	}
	
	public Unary(Lop input1, Lop input2, OperationTypes op, DataType dt, ValueType vt) {
		super(Lop.Type.UNARY, dt, vt);
		init(input1, input2, op, dt, vt, ExecType.MR);
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

		// By definition, this lop should not break alignment
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;

		if ( et == ExecType.MR ) {
			/*
			 * This lop CAN NOT be executed in PARTITION, SORT, CM_COV, and COMBINE
			 * jobs MMCJ: only in mapper.
			 */
			lps.addCompatibility(JobType.ANY);
			lps.removeNonPiggybackableJobs();
			lps.removeCompatibility(JobType.CM_COV); // CM_COV allows only reducer instructions but this is MapOrReduce. TODO: piggybacking should be updated to take this extra constraint.
			lps.removeCompatibility(JobType.TRANSFORM);
			this.lps.setProperties(inputs, et, ExecLocation.MapOrReduce, breaksAlignment, aligner, definesMRJob);
		}
		else {
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties(inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
		}
	}

	/**
	 * Constructor to perform a unary operation with 1 input.
	 * 
	 * @param input1
	 * @param op
	 */
	public Unary(Lop input1, OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.UNARY, dt, vt);
		init(input1, op, dt, vt, et);
	}
	
	public Unary(Lop input1, OperationTypes op, DataType dt, ValueType vt) {
		super(Lop.Type.UNARY, dt, vt);
		init(input1, op, dt, vt, ExecType.MR);
	}
	
	private ExecType forceExecType(OperationTypes op, ExecType et) {
		if ( op == OperationTypes.INVERSE )
			return ExecType.CP;
		return et;
	}
	private void init(Lop input1, OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		operation = op;

		et = forceExecType(op, et);
		
		valInput = null;

		this.addInput(input1);
		input1.addOutput(this);

		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;

		if ( et == ExecType.MR ) {
			/*
			 * This lop can be executed in all jobs except for PARTITION. MMCJ: only
			 * in mapper. GroupedAgg: only in reducer.
			 */
			lps.addCompatibility(JobType.ANY);
			lps.removeNonPiggybackableJobs();
			lps.removeCompatibility(JobType.CM_COV); // CM_COV allows only reducer instructions but this is MapOrReduce. TODO: piggybacking should be updated to take this extra constraint.
			lps.removeCompatibility(JobType.TRANSFORM);
			this.lps.setProperties(inputs, et, ExecLocation.MapOrReduce, breaksAlignment, aligner, definesMRJob);
		}
		else {
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties(inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
		}
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

	private String getOpcode() throws LopsException {
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
			
		case INVERSE:
			return "inverse";
			
		case MR_IQM:
			return "qpick";

		case SPROP:
			return "sprop";
			
		case SIGMOID:
			return "sigmoid";
		
		case SELP:
			return "sel+";
		
		default:
			throw new LopsException(this.printErrorLocation() + 
					"Instruction not defined for Unary operation: " + operation);
		}
	}
	public String getInstructions(String input1, String output) 
		throws LopsException 
	{
		// Unary operators with one input
		if (this.getInputs().size() == 1) {
			
			StringBuilder sb = new StringBuilder();
			sb.append( getExecType() );
			sb.append( Lop.OPERAND_DELIMITOR );
			sb.append( getOpcode() );
			sb.append( OPERAND_DELIMITOR );
			sb.append( getInputs().get(0).prepInputOperand(input1));
			sb.append( OPERAND_DELIMITOR );
			sb.append( this.prepOutputOperand(output));
			
			return sb.toString();

		} else {
			throw new LopsException(this.printErrorLocation() + "Invalid number of operands ("
					+ this.getInputs().size() + ") for an Unary opration: "
					+ operation);
		}
	}
	
	@Override
	public String getInstructions(int input_index, int output_index)
			throws LopsException {
		return getInstructions(""+input_index, ""+output_index);
	}

	@Override
	public String getInstructions(String input1, String input2, String output) 
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		
		if ( getInputs().get(0).getDataType() == DataType.SCALAR ) {
			sb.append( getInputs().get(0).prepScalarInputOperand(getExecType()));
		}
		else {
			sb.append( getInputs().get(0).prepInputOperand(input1));
		}
		sb.append( OPERAND_DELIMITOR );
		
		if ( getInputs().get(1).getDataType() == DataType.SCALAR ) {
			sb.append( getInputs().get(1).prepScalarInputOperand(getExecType()));
		}
		else {
			sb.append( getInputs().get(1).prepInputOperand(input2));
		}
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();
	}
	
	@Override
	public String getInstructions(int inputIndex1, int inputIndex2,
			int outputIndex) throws LopsException {
		if (this.getInputs().size() == 2) {
			// Unary operators with two inputs
			// Determine the correct operation, depending on the scalar input
			Lop linput1 = getInputs().get(0);
			Lop linput2 = getInputs().get(1);
			
			int scalarIndex = -1, matrixIndex = -1;
			String matrixLabel= null;
			if( linput1.getDataType() == DataType.MATRIX ) {
				// inputIndex1 is matrix, and inputIndex2 is scalar
				scalarIndex = 1;
				matrixLabel = String.valueOf(inputIndex1);
			}
			else {
				// inputIndex2 is matrix, and inputIndex1 is scalar
				scalarIndex = 0;
				matrixLabel = String.valueOf(inputIndex2); 
				
				// when the first operand is a scalar, setup the operation type accordingly
				if (operation == OperationTypes.SUBTRACT)
					operation = OperationTypes.SUBTRACTRIGHT;
				else if (operation == OperationTypes.DIVIDE)
					operation = OperationTypes.Over;
			}
			matrixIndex = 1-scalarIndex;

			// Prepare the instruction
			StringBuilder sb = new StringBuilder();
			sb.append( getExecType() );
			sb.append( Lop.OPERAND_DELIMITOR );
			sb.append( getOpcode() );
			sb.append( OPERAND_DELIMITOR );
			
			if(  operation == OperationTypes.INTDIV || operation == OperationTypes.MODULUS || 
				 operation == OperationTypes.POW || 
				 operation == OperationTypes.GREATER_THAN || operation == OperationTypes.GREATER_THAN_OR_EQUALS ||
				 operation == OperationTypes.LESS_THAN || operation == OperationTypes.LESS_THAN_OR_EQUALS ||
				 operation == OperationTypes.EQUALS || operation == OperationTypes.NOT_EQUALS )
			{
				//TODO discuss w/ Shirish: we should consolidate the other operations (see ScalarInstruction.parseInstruction / BinaryCPInstruction.getScalarOperator)
				//append both operands
				sb.append( (linput1.getDataType()==DataType.MATRIX? linput1.prepInputOperand(String.valueOf(inputIndex1)) : linput1.prepScalarInputOperand(getExecType())) );
				sb.append( OPERAND_DELIMITOR );
				sb.append( (linput2.getDataType()==DataType.MATRIX? linput2.prepInputOperand(String.valueOf(inputIndex2)) : linput2.prepScalarInputOperand(getExecType())) );
				sb.append( OPERAND_DELIMITOR );	
			}
			else
			{
				// append the matrix operand
				sb.append( getInputs().get(matrixIndex).prepInputOperand(matrixLabel));
				sb.append( OPERAND_DELIMITOR );
				
				// append the scalar operand
				sb.append( getInputs().get(scalarIndex).prepScalarInputOperand(getExecType()));
				sb.append( OPERAND_DELIMITOR );
			}
			sb.append( this.prepOutputOperand(outputIndex+""));
			
			return sb.toString();
			
		} else {
			throw new LopsException(this.printErrorLocation() + "Invalid number of operands ("
					+ this.getInputs().size() + ") for an Unary opration: "
					+ operation);
		}
	}
}
