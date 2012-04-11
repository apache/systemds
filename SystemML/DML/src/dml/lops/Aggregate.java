package dml.lops;

import dml.lops.LopProperties.ExecLocation;
import dml.lops.LopProperties.ExecType;
import dml.lops.PartialAggregate.CorrectionLocationType;
import dml.lops.compile.JobType;
import dml.parser.Expression.*;
import dml.utils.LopsException;

/**
 * Lop to represent an aggregation.
 * It is used in rowsum, colsum, etc. 
 * @author aghoting
 */

public class Aggregate extends Lops 
{
	
	/** Aggregate operation types **/
	
	public enum OperationTypes {Sum,Product,Min,Max,Trace,DiagM2V,KahanSum,KahanTrace,Mean,MaxIndex};	
	OperationTypes operation;
 
	private boolean isCorrectionUsed = false;
	private CorrectionLocationType correctionLocation = CorrectionLocationType.INVALID;

	/**
	 * @param input - input lop
	 * @param op - operation type
	 */
	public Aggregate(Lops input, Aggregate.OperationTypes op, DataType dt, ValueType vt ) {
		super(Lops.Type.Aggregate, dt, vt);
		init ( input, op, dt, vt, ExecType.MR );
	}
	
	public Aggregate(Lops input, Aggregate.OperationTypes op, DataType dt, ValueType vt, ExecType et ) {
		super(Lops.Type.Aggregate, dt, vt);
		init ( input, op, dt, vt, et );
	}
	
	private void init (Lops input, Aggregate.OperationTypes op, DataType dt, ValueType vt, ExecType et ) {
		operation = op;	
		this.addInput(input);
		input.addOutput(this);
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		if ( et == ExecType.MR ) {
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.RAND);
			lps.addCompatibility(JobType.REBLOCK_BINARY);
			lps.addCompatibility(JobType.REBLOCK_TEXT);
			this.lps.setProperties( et, ExecLocation.Reduce, breaksAlignment, aligner, definesMRJob );
		}
		else {
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties( et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		}
	}
	
	// this function must be invoked during hop-to-lop translation
	public void setupCorrectionLocation(CorrectionLocationType loc) {
		if ( operation == OperationTypes.KahanSum || operation == OperationTypes.KahanTrace || operation == OperationTypes.Mean ) {
			isCorrectionUsed = true;
			correctionLocation = loc;
		}
	}
	
	/**
	 * for debugging purposes. 
	 */
	
	public String toString()
	{
		return "Operation: " + operation;		
	}

	/**
	 * method to get operation type
	 * @return
	 */
	 
	public OperationTypes getOperationType()
	{
		return operation;
	}
	
	
	private String getOpcode() {
		switch(operation) {
		case Sum: 
		case Trace: 
			return "a+"; 
		case Mean: 
			return "amean"; 
		case Product: 
			return "a*"; 
		case Min: 
			return "amin"; 
		case Max: 
			return "amax"; 
		case MaxIndex:
			return "arimax";
			
		case KahanSum:
		case KahanTrace: 
			return "ak+"; 
		default:
			throw new UnsupportedOperationException("Instruction is not defined for Aggregate operation: " + operation);
		}
	}
	
	@Override
	public String getInstructions(String input1, String output) throws LopsException {
		String opcode = getOpcode(); 
		String inst = getExecType() + OPERAND_DELIMITOR + opcode + OPERAND_DELIMITOR + 
		        input1 + DATATYPE_PREFIX + getInputs().get(0).get_dataType() + VALUETYPE_PREFIX + getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
		        output + DATATYPE_PREFIX + get_dataType() + VALUETYPE_PREFIX + get_valueType() ;
		return inst;
	}
	
	@Override
	public String getInstructions(int input_index, int output_index) throws LopsException
	{
		boolean isCorrectionApplicable = false;
		
		String opcode = getOpcode(); 
		if (operation == OperationTypes.Mean || operation == OperationTypes.KahanSum || operation == OperationTypes.KahanTrace ) 
			isCorrectionApplicable = true;
		
		String inst = getExecType() + OPERAND_DELIMITOR + opcode + OPERAND_DELIMITOR + 
		        input_index + DATATYPE_PREFIX + getInputs().get(0).get_dataType() + VALUETYPE_PREFIX + getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
		        output_index + DATATYPE_PREFIX + get_dataType() + VALUETYPE_PREFIX + get_valueType() ;
		
		if ( isCorrectionApplicable )
			// add correction information to the instruction
			inst += OPERAND_DELIMITOR + isCorrectionUsed + OPERAND_DELIMITOR + correctionLocation;
		
		return inst;
	}

 
 
}
