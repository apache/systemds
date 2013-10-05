/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum OperationTypes {
		ADD, SUBTRACT, SUBTRACTRIGHT, MULTIPLY, DIVIDE, MODULUS, POW, LOG, MAX, MIN, NOT, ABS, SIN, COS, TAN, SQRT, EXP, Over, LESS_THAN, LESS_THAN_OR_EQUALS, GREATER_THAN, GREATER_THAN_OR_EQUALS, EQUALS, NOT_EQUALS, ROUND, NOTSUPPORTED
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

		if (input1.get_dataType() == DataType.MATRIX)
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
			lps.removeCompatibility(JobType.PARTITION);
			lps.removeCompatibility(JobType.SORT);
			lps.removeCompatibility(JobType.CM_COV);
			lps.removeCompatibility(JobType.COMBINE);
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
	
	private void init(Lop input1, OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		operation = op;

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
			lps.removeCompatibility(JobType.PARTITION);
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
			return " Operation: " + operation + " " + "Label: "
					+ valInput.getOutputParameters().getLabel()
					+ " input types " + this.getInputs().get(0).toString()
					+ " " + this.getInputs().get(1).toString();
		else
			return " Operation: " + operation + " " + "Label: N/A";
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
		case SQRT:
			return "sqrt";
		case EXP:
			return "exp";
		
		case LOG:
			return "log";
		
		case ROUND:
			return "round";

		case ADD:
			return "+";

		case SUBTRACT:
			return "-";

		case SUBTRACTRIGHT:
			return "s-r";

		case MULTIPLY:
			return "*";

		case DIVIDE:
			return "/";

		case MODULUS:
			return "%%";			
			
		case Over:
			return "so";

		case POW:
			return "^";

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
			sb.append( input1 );
			sb.append( DATATYPE_PREFIX );
			sb.append( getInputs().get(0).get_dataType() );
			sb.append( VALUETYPE_PREFIX );
			sb.append( getInputs().get(0).get_valueType() );
			sb.append( OPERAND_DELIMITOR );
			sb.append( output );
			sb.append( DATATYPE_PREFIX );
			sb.append( get_dataType() );
			sb.append( VALUETYPE_PREFIX );
			sb.append( get_valueType() );
			
			return sb.toString();

		} else {
			throw new LopsException(this.printErrorLocation() + "Invalid number of operands ("
					+ this.getInputs().size() + ") for an Unary opration: "
					+ operation);
		}
	}
	
	public String getInstructions(String input1, String input2, String output) 
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( input1 );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(0).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(0).get_valueType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( input2 );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(1).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(1).get_valueType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( output );
		sb.append( DATATYPE_PREFIX );
		sb.append( get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( get_valueType() );
		
		return sb.toString();
	}
	
	public String getInstructions(int input_index, int output_index)
			throws LopsException {
		return getInstructions(""+input_index, ""+output_index);
	}

	@Override
	public String getInstructions(int inputIndex1, int inputIndex2,
			int outputIndex) throws LopsException {
		if (this.getInputs().size() == 2) {
			// Unary operators with two inputs

			// Determine which of the two inputs is a scalar
			int scalarIndex = -1;

			if (this.getInputs().get(0).get_dataType() == DataType.MATRIX)
				// inputIndex1 is matrix, and inputIndex2 is scalar
				scalarIndex = 1;

			else {
				// inputIndex2 is matrix, and inputIndex1 is scalar
				scalarIndex = 0;
				if (operation == OperationTypes.SUBTRACT)
					operation = OperationTypes.SUBTRACTRIGHT;
				else if (operation == OperationTypes.DIVIDE)
					operation = OperationTypes.Over;
			}

			/**
			 * get label for scalar input.
			 */

			String valueLabel = this.getInputs().get(scalarIndex)
					.getOutputParameters().getLabel();

			/**
			 * if it is a literal, copy val, else surround with the label with
			 * ## symbols. these will be replaced at runtime.
			 */
			String valueString;
			if (this.getInputs().get(scalarIndex).getExecLocation() == ExecLocation.Data
					&& ((Data) this.getInputs().get(scalarIndex)).isLiteral())
				valueString = "" + valueLabel;
			else
				valueString = "##" + valueLabel + "##";

			if (scalarIndex == 1) {
				// second input is the scalar
				return getInstructions(""+inputIndex1, valueString, ""+outputIndex);
			} else {
				// first input is the scalar
				return getInstructions(""+inputIndex2, valueString, ""+outputIndex);
			}
		} else {
			throw new LopsException(this.printErrorLocation() + "Invalid number of operands ("
					+ this.getInputs().size() + ") for an Unary opration: "
					+ operation);
		}
	}
}
