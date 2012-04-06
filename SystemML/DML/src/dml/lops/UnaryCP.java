package dml.lops;

import dml.lops.LopProperties.ExecLocation;
import dml.lops.LopProperties.ExecType;
import dml.lops.compile.JobType;
import dml.parser.Expression.DataType;
import dml.utils.LopsException;
import dml.parser.Expression.*;

/**
 * Lop to perform unary scalar operations. Example a = !b
 * 
 * @author aghoting
 */

public class UnaryCP extends Lops {
	public enum OperationTypes {
		NOT, ABS, SIN, COS, TAN, SQRT, LOG, EXP, CAST_AS_SCALAR, PRINT, NROW, NCOL, LENGTH, ROUND, SPEARMANHELPER, PRINT2, NOTSUPPORTED
	};

	OperationTypes operation;

	/**
	 * Constructor to perform a scalar operation
	 * 
	 * @param input
	 * @param op
	 */

	public UnaryCP(Lops input, OperationTypes op, DataType dt, ValueType vt) {
		super(Lops.Type.UnaryCP, dt, vt);
		operation = op;
		this.addInput(input);
		input.addOutput(this);

		/*
		 * This lop is executed in control program.
		 */
		boolean breaksAlignment = false; // this does not carry any information
											// for this lop
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.INVALID);
		this.lps.setProperties(ExecType.CP, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
	}

	@Override
	public String toString() {

		return "Operation: " + operation;

	}

	@Override
	public String getInstructions(String input, String output)
			throws LopsException {
		String opString = new String(getExecType() + Lops.OPERAND_DELIMITOR);
		ValueType vtype = this.getInputs().get(0).get_valueType();

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

		// CAST_AS_SCALAR, NROW, NCOL, LENGTH builtins take matrix as the input
		// and produces a scalar
		case CAST_AS_SCALAR:
			opString += "assignvarwithfile";
			break;
		case NROW:
			opString += "nrow";
			vtype = ValueType.STRING;
			break;
		case NCOL:
			opString += "ncol";
			vtype = ValueType.STRING;
			break;
		case LENGTH:
			opString += "length";
			vtype = ValueType.STRING;
			break;

		case SPEARMANHELPER:
			opString += "spearmanhelper";
			vtype = ValueType.STRING;
			break;

		default:
			throw new LopsException(
					"Instruction not defined for UnaryScalar opration: "
							+ operation);
		}

		String inst = new String("");
		inst += opString + OPERAND_DELIMITOR + input + DATATYPE_PREFIX + getInputs().get(0).get_dataType() + VALUETYPE_PREFIX + vtype
				+ OPERAND_DELIMITOR + output + DATATYPE_PREFIX + get_dataType() + VALUETYPE_PREFIX + this.get_valueType();
		return inst;

	}
}
