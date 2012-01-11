package dml.lops;

import dml.utils.LopsException;
import dml.lops.LopProperties.ExecLocation;
import dml.lops.compile.JobType;
import dml.parser.Expression.*;

/**
 * Lop to perform a partial aggregation. It was introduced to do some initial
 * aggregation operations on blocks in the mapper/reducer.
 * 
 * @author aghoting
 */

public class PartialAggregate extends Lops {

	public enum DirectionTypes {
		RowCol, Row, Col
	};

	Aggregate.OperationTypes operation;
	DirectionTypes direction;

	/**
	 * Constructor to setup a partial aggregate operation.
	 * 
	 * @param input
	 * @param op
	 * @throws LopsException
	 */

	public PartialAggregate(
			Lops input,
			Aggregate.OperationTypes op,
			PartialAggregate.DirectionTypes direct, DataType dt, ValueType vt)
			throws LopsException {
		super(Lops.Type.PartialAggregate, dt, vt);
		operation = op;
		direction = direct;
		this.addInput(input);
		input.addOutput(this);

		/*
		 * This lop CAN NOT be executed in PARTITION, SORT, STANDALONE MMCJ:
		 * only in mapper.
		 */
		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		lps.addCompatibility(JobType.GMR);
		lps.addCompatibility(JobType.RAND);
		lps.addCompatibility(JobType.REBLOCK_BINARY);
		lps.addCompatibility(JobType.REBLOCK_TEXT);
		lps.addCompatibility(JobType.MMCJ);
		lps.addCompatibility(JobType.MMRJ);
		this.lps.setProperties(ExecLocation.Map, breaksAlignment, aligner, definesMRJob);
	}

	/**
	 * This method computes the location of "correction" terms in the output
	 * produced by PartialAgg instruction.
	 * 
	 * When computing the stable sum, "correction" refers to the compensation as
	 * defined by the original Kahan algorithm. When computing the stable mean,
	 * "correction" refers to two extra values (the running mean, count)
	 * produced by each Mapper i.e., by each PartialAgg instruction.
	 * 
	 * This method is invoked during hop-to-lop translation, while creating the
	 * corresponding Aggregate lop
	 * 
	 * Computed information is encoded in the PartialAgg instruction so that the
	 * appropriate aggregate operator is used at runtime (see:
	 * dml.runtime.matrix.operator.AggregateOperator.java and dml.runtime.matrix)
	 */
	public byte getCorrectionLocaion() throws LopsException {

		byte loc;

		switch (operation) {
		case KahanSum:
		case KahanTrace:
			switch (direction) {
			case Col:
				// colSums: corrections will be present as a last row in the
				// result
				loc = 1;
				break;
			case Row:
			case RowCol:
				// rowSums, sum: corrections will be present as a last column in
				// the result
				loc = 2;
				break;
			default:
				throw new LopsException(
						"getCorrectionLocaion():: Unknown aggregarte direction - "
								+ direction);
			}
			break;

		case Mean:
			// Computation of stable mean requires each mapper to output both
			// the running mean as well as the count
			switch (direction) {
			case Col:
				// colMeans: last row is correction 2nd last is count
				loc = 3;
				break;
			case Row:
			case RowCol:
				// rowMeans, mean: last column is correction 2nd last is count
				loc = 4;
				break;
			default:
				throw new LopsException(
						"getCorrectionLocaion():: Unknown aggregarte direction - "
								+ direction);
			}
			break;

		default:
			// this function is valid only when kahanSum or stableMean is
			// computed
			loc = 0;
		}
		return loc;
	}

	public void setDimensionsBasedOnDirection(long dim1, long dim2,
			int rowsPerBlock, int colsPerBlock) throws LopsException {
		if (direction == DirectionTypes.Row)
			outParams.setDimensions(dim1, 1, rowsPerBlock, colsPerBlock);
		else if (direction == DirectionTypes.Col)
			outParams.setDimensions(1, dim2, rowsPerBlock, colsPerBlock);
		else if (direction == DirectionTypes.RowCol)
			outParams.setDimensions(1, 1, rowsPerBlock, colsPerBlock);
		else
			throw new LopsException("Unknown aggregate direction " + direction);
	}

	public String toString() {
		return "Partial Aggregate " + operation;
	}

	@Override
	public String getInstructions(int input_index, int output_index)
			throws LopsException {

		/** Sum row col **/
		String opString = new String("");

		if (operation == Aggregate.OperationTypes.Sum
				&& direction == DirectionTypes.RowCol) {
			opString += "ua+";
		} else if (operation == Aggregate.OperationTypes.Sum
				&& direction == DirectionTypes.Row) {
			opString += "uar+";
		} else if (operation == Aggregate.OperationTypes.Sum
				&& direction == DirectionTypes.Col) {
			opString += "uac+";
		}

		if (operation == Aggregate.OperationTypes.Mean
				&& direction == DirectionTypes.RowCol) {
			opString += "uamean";
		} else if (operation == Aggregate.OperationTypes.Mean
				&& direction == DirectionTypes.Row) {
			opString += "uarmean";
		} else if (operation == Aggregate.OperationTypes.Mean
				&& direction == DirectionTypes.Col) {
			opString += "uacmean";
		}

		// instructions that use kahanSum are similar to ua+,uar+,uac+
		// except that they also produce correction values along with partial
		// sums.
		else if (operation == Aggregate.OperationTypes.KahanSum
				&& direction == DirectionTypes.RowCol) {
			opString += "uak+";
		} else if (operation == Aggregate.OperationTypes.KahanSum
				&& direction == DirectionTypes.Row) {
			opString += "uark+";
		} else if (operation == Aggregate.OperationTypes.KahanSum
				&& direction == DirectionTypes.Col) {
			opString += "uack+";
		}

		else if (operation == Aggregate.OperationTypes.Product
				&& direction == DirectionTypes.RowCol) {
			opString += "ua*";
		}

		else if (operation == Aggregate.OperationTypes.Max
				&& direction == DirectionTypes.RowCol) {
			opString += "uamax";
		} else if (operation == Aggregate.OperationTypes.Max
				&& direction == DirectionTypes.Row) {
			opString += "uarmax";
		} else if (operation == Aggregate.OperationTypes.Max
				&& direction == DirectionTypes.Col) {
			opString += "uacmax";
		}

		else if (operation == Aggregate.OperationTypes.Min
				&& direction == DirectionTypes.RowCol) {
			opString += "uamin";
		} else if (operation == Aggregate.OperationTypes.Min
				&& direction == DirectionTypes.Row) {
			opString += "uarmin";
		} else if (operation == Aggregate.OperationTypes.Min
				&& direction == DirectionTypes.Col) {
			opString += "uacmin";
		}

		else if (operation == Aggregate.OperationTypes.Trace
				&& direction == DirectionTypes.RowCol) {
			opString += "uatrace";
		} else if (operation == Aggregate.OperationTypes.KahanTrace
				&& direction == DirectionTypes.RowCol) {
			opString += "uaktrace";
		} else if (operation == Aggregate.OperationTypes.DiagM2V
				&& direction == DirectionTypes.Col) {
			opString += "rdiagM2V";
		} else {
			throw new UnsupportedOperationException(
					"Instruction is not defined for PartialAggregate operation "
							+ operation);
		}

		String inst = new String("");
		inst += opString + OPERAND_DELIMITOR + input_index + VALUETYPE_PREFIX
				+ this.getInputs().get(0).get_valueType() + OPERAND_DELIMITOR
				+ output_index + VALUETYPE_PREFIX + this.get_valueType();

		return inst;
	}

}
