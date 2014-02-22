/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum OperationTypes {
		NOT, ABS, SIN, COS, TAN, ASIN, ACOS, ATAN, SQRT, LOG, EXP, CAST_AS_SCALAR, CAST_AS_MATRIX, CAST_AS_DOUBLE, CAST_AS_INT, CAST_AS_BOOLEAN, PRINT, NROW, NCOL, LENGTH, ROUND, PRINT2, NOTSUPPORTED
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

	@Override
	public String getInstructions(String input, String output)
			throws LopsException {
		String opString = new String(getExecType() + Lop.OPERAND_DELIMITOR);

		switch (operation) {
		case NOT:
			opString += "!";
			break;

		case ABS:
			opString += "abs";
			break;

		case SIN:
			opString += "sin";
			break;

		case COS:
			opString += "cos";
			break;

		case TAN:
			opString += "tan";
			break;

		case ASIN:
			opString += "asin";
			break;

		case ACOS:
			opString += "acos";
			break;

		case ATAN:
			opString += "atan";
			break;

		case SQRT:
			opString += "sqrt";
			break;

		case LOG:
			opString += "log";
			break;

		case ROUND:
			opString += "round";
			break;

		case EXP:
			opString += "exp";
			break;

		case PRINT:
			opString += "print";
			break;

		case PRINT2:
			opString += "print2";
			break;

		case CAST_AS_MATRIX:
			opString += CAST_AS_MATRIX_OPCODE;
			break;	
			
		// CAST_AS_SCALAR, NROW, NCOL, LENGTH builtins take matrix as the input
		// and produces a scalar
		case CAST_AS_SCALAR:
			opString += CAST_AS_SCALAR_OPCODE; //old opcode "assignvarwithfile";
			break;

		case CAST_AS_DOUBLE:
			opString += CAST_AS_DOUBLE_OPCODE; 
			break;

		case CAST_AS_INT:
			opString += CAST_AS_INT_OPCODE; 
			break;

		case CAST_AS_BOOLEAN:
			opString += CAST_AS_BOOLEAN_OPCODE; 
			break;

		case NROW:
			opString += "nrow";
			break;
		case NCOL:
			opString += "ncol";
			break;
		case LENGTH:
			opString += "length";
			break;

		default:
			throw new LopsException(this.printErrorLocation() + 
					"Instruction not defined for UnaryScalar operation: "
							+ operation);
		}

		StringBuilder sb = new StringBuilder();
		sb.append( opString );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append(getInputs().get(0).prepScalarInputOperand(getExecType()));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();

	}
}
