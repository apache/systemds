package dml.lops;

import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;
import dml.utils.LopsException;
import dml.lops.LopProperties.ExecLocation;
import dml.lops.compile.JobType;

/**
 * Lop to perform following operations: with one operand -- NOT(A), ABS(A),
 * SQRT(A), LOG(A) with two operands where one of them is a scalar -- H=H*i,
 * H=H*5, EXP(A,2), LOG(A,2)
 * 
 * @author aghoting
 */

public class Unary extends Lops {
	public enum OperationTypes {
		ADD, SUBTRACT, SUBTRACTRIGHT, MULTIPLY, DIVIDE, POW, LOG, MAX, MIN, NOT, ABS, SIN, COS, TAN, SQRT, EXP, Over, LESS_THAN, LESS_THAN_OR_EQUALS, GREATER_THAN, GREATER_THAN_OR_EQUALS, EQUALS, NOT_EQUALS, ROUND, NOTSUPPORTED
	};

	OperationTypes operation;

	Lops valInput;

	/**
	 * Constructor to perform a unary operation with 2 inputs
	 * 
	 * @param input
	 * @param op
	 */

	public Unary(Lops input1, Lops input2, OperationTypes op, DataType dt,
			ValueType vt) {
		super(Lops.Type.UNARY, dt, vt);
		operation = op;

		if (input1.get_dataType() == DataType.MATRIX)
			valInput = input2;
		else
			valInput = input1;

		this.addInput(input1);
		input1.addOutput(this);

		this.addInput(input2);
		input2.addOutput(this);

		/*
		 * This lop CAN NOT be executed in PARTITION, SORT, CM_COV, and COMBINE
		 * jobs MMCJ: only in mapper.
		 */
		lps.addCompatibility(JobType.ANY);
		lps.removeCompatibility(JobType.PARTITION);
		lps.removeCompatibility(JobType.SORT);
		lps.removeCompatibility(JobType.CM_COV);
		lps.removeCompatibility(JobType.COMBINE);

		// By definition, this lop should not break alignment
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;

		this.lps.setProperties(ExecLocation.MapOrReduce, breaksAlignment, aligner, definesMRJob);
	}

	/**
	 * Constructor to perform a unary operation with 1 input.
	 * 
	 * @param input1
	 * @param op
	 */

	public Unary(Lops input1, OperationTypes op, DataType dt, ValueType vt) {
		super(Lops.Type.UNARY, dt, vt);
		operation = op;

		valInput = null;

		this.addInput(input1);
		input1.addOutput(this);

		/*
		 * This lop can be executed in all jobs except for PARTITION. MMCJ: only
		 * in mapper. GroupedAgg: only in reducer.
		 */
		lps.addCompatibility(JobType.ANY);
		lps.removeCompatibility(JobType.PARTITION);

		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;

		this.lps.setProperties(ExecLocation.MapOrReduce, breaksAlignment, aligner, definesMRJob);
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

	public String getInstructions(int input_index, int output_index)
			throws LopsException {
		String opString = new String("");

		if (this.getInputs().size() == 1) {
			// Unary operators with one input
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
			case EXP:
				opString += "exp";
				break;
			case LOG:
				opString += "log";
				break;
			case ROUND:
				opString += "round";
				break;

			default:
				throw new LopsException(
						"Instruction not defined for Unary operation: "
								+ operation);
			}
			String inst = new String("");
			inst += opString + OPERAND_DELIMITOR + input_index
					+ VALUETYPE_PREFIX
					+ this.getInputs().get(0).get_valueType()
					+ OPERAND_DELIMITOR + output_index + VALUETYPE_PREFIX
					+ this.get_valueType();

			return inst;

		} else {
			throw new LopsException("Invalid number of operands ("
					+ this.getInputs().size() + ") for an Unary opration: "
					+ operation);
		}
	}

	@Override
	public String getInstructions(int inputIndex1, int inputIndex2,
			int outputIndex) throws LopsException {
		String valueString;
		String opString = new String("");

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
			if (this.getInputs().get(scalarIndex).getExecLocation() == ExecLocation.Data
					&& ((Data) this.getInputs().get(scalarIndex)).isLiteral())
				valueString = "" + valueLabel;
			else
				valueString = "##" + valueLabel + "##";

			switch (operation) {
			case ADD:
				opString += "s+";
				break;

			case SUBTRACT:
				opString += "s-";
				break;

			case SUBTRACTRIGHT:
				opString += "s-r";
				break;

			case MULTIPLY:
				opString += "s*";
				break;

			case DIVIDE:
				opString += "s/";
				break;

			case Over:
				opString += "so";
				break;

			case POW:
				opString += "s^";
				break;

			case GREATER_THAN:
				opString += "s>";
				break;

			case GREATER_THAN_OR_EQUALS:
				opString += "s>=";
				break;

			case LESS_THAN:
				opString += "s<";
				break;

			case LESS_THAN_OR_EQUALS:
				opString += "s<=";
				break;

			case EQUALS:
				opString += "s==";
				break;

			case NOT_EQUALS:
				opString += "s!=";
				break;

			case LOG:
				// Unlike other Unary Lops, Scalar LOG (slog) is translated to
				// UnaryInstruction as opposed ScalarInstructions
				opString += "slog";
				break;

			case MAX:
				// TODO: does it make sense to say MAX(A,2) ?
				opString += "smax";
				break;

			case MIN:
				// TODO: does it make sense to say MIN(A,2) ?
				opString += "smin";
				break;

			default:
				throw new LopsException(
						"Instruction not defined for Unary opration: "
								+ operation);
			}

			String inst = new String("");
			if (scalarIndex == 1) {
				// second input is the scalar
				inst += opString + OPERAND_DELIMITOR + inputIndex1
						+ VALUETYPE_PREFIX
						+ this.getInputs().get(0).get_valueType()
						+ OPERAND_DELIMITOR + valueString + VALUETYPE_PREFIX
						+ this.getInputs().get(1).get_valueType()
						+ OPERAND_DELIMITOR + outputIndex + VALUETYPE_PREFIX
						+ this.get_valueType();
			} else {
				// first input is the scalar
				inst += opString + OPERAND_DELIMITOR + inputIndex2
						+ VALUETYPE_PREFIX
						+ this.getInputs().get(1).get_valueType()
						+ OPERAND_DELIMITOR + valueString + VALUETYPE_PREFIX
						+ this.getInputs().get(0).get_valueType()
						+ OPERAND_DELIMITOR + outputIndex + VALUETYPE_PREFIX
						+ this.get_valueType();
			}

			return inst;
		} else {
			throw new LopsException("Invalid number of operands ("
					+ this.getInputs().size() + ") for an Unary opration: "
					+ operation);
		}
	}
}
