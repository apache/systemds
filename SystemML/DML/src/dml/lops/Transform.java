package dml.lops;

import dml.utils.LopsException;
import dml.lops.LopProperties.ExecLocation;
import dml.lops.compile.JobType;
import dml.parser.Expression.*;

/*
 * Lop to perform transpose/vector to diag operations
 * This lop can change the keys and hence break alignment.
 */

public class Transform extends Lops
{
	
	public enum OperationTypes {Transpose,VectortoDiagMatrix};
	
	OperationTypes operation = null;
	
	/**
	 * Constructor when we have one input.
	 * @param input
	 * @param op
	 */

	public Transform(Lops input, Transform.OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lops.Type.Transform, dt, vt);		
		operation = op;
 
		this.addInput(input);
		input.addOutput(this);

		/*
		 *  This lop CAN NOT be executed in PARTITION, SORT, STANDALONE
		 *  MMCJ: only in mapper.
		 */
		lps.addCompatibility(JobType.GMR);
		lps.addCompatibility(JobType.RAND);
		lps.addCompatibility(JobType.REBLOCK_BINARY);
		lps.addCompatibility(JobType.REBLOCK_TEXT);
		lps.addCompatibility(JobType.MMCJ);
		lps.addCompatibility(JobType.MMRJ);

		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		this.lps.setProperties( ExecLocation.MapOrReduce, breaksAlignment, aligner, definesMRJob );
	}
	
	

	@Override
	public String toString() {

		return " Operation: " + operation;
	}

	/**
	 * method to get operation type
	 * @return
	 */
	 
	public OperationTypes getOperationType()
	{
		return operation;
	}


	@Override 
	public String getInstructions(int input_index, int output_index) throws LopsException
	{
		String opString = new String("");
		switch(operation) {
		case Transpose:
			// Transpose a matrix
			opString += "r'"; break;
		
		case VectortoDiagMatrix:
			// Transform a vector into a diagonal matrix
			opString += "rdiagV2M"; break;
		
		default:
			throw new UnsupportedOperationException("Instruction is not defined for Transform operation " + operation);
				
		}
		
		String inst = new String("");
		inst += opString + OPERAND_DELIMITOR + 
				input_index + VALUETYPE_PREFIX + this.getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
		        output_index + VALUETYPE_PREFIX + this.get_valueType() ;
		return inst;
	}


 
 
}