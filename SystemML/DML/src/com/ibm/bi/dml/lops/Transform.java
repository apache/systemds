package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.*;
import com.ibm.bi.dml.utils.LopsException;


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

	public Transform(Lops input, Transform.OperationTypes op, DataType dt, ValueType vt, ExecType et) 
	{
		super(Lops.Type.Transform, dt, vt);		
		init(input, op, dt, vt, et);
	}
	
	public Transform(Lops input, Transform.OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lops.Type.Transform, dt, vt);		
		init(input, op, dt, vt, ExecType.MR);
	}

	private void init (Lops input, Transform.OperationTypes op, DataType dt, ValueType vt, ExecType et) 
	{
		operation = op;
 
		this.addInput(input);
		input.addOutput(this);

		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		if ( et == ExecType.MR ) {
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
			this.lps.setProperties( et, ExecLocation.MapOrReduce, breaksAlignment, aligner, definesMRJob );
		}
		else {
			// <code>breaksAlignment</code> is not meaningful when <code>Transform</code> executes in CP. 
			breaksAlignment = false;
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties( et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		}
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

	private String getOpcode() {
		switch(operation) {
		case Transpose:
			// Transpose a matrix
			return "r'";
		
		case VectortoDiagMatrix:
			// Transform a vector into a diagonal matrix
			return "rdiagV2M";
		
		default:
			throw new UnsupportedOperationException("Instruction is not defined for Transform operation " + operation);
				
		}
	}
	
	@Override
	public String getInstructions(String input1, String output) throws LopsException {
		String inst = getExecType() + OPERAND_DELIMITOR + getOpcode() + OPERAND_DELIMITOR + 
		        input1 + DATATYPE_PREFIX + getInputs().get(0).get_dataType() + VALUETYPE_PREFIX + getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
		        output + DATATYPE_PREFIX + get_dataType() + VALUETYPE_PREFIX + get_valueType() ;
		return inst;
	}
	

	@Override 
	public String getInstructions(int input_index, int output_index) throws LopsException
	{
		String inst = getExecType() + OPERAND_DELIMITOR + getOpcode() 
				+ OPERAND_DELIMITOR + input_index + DATATYPE_PREFIX + getInputs().get(0).get_dataType() + VALUETYPE_PREFIX + this.getInputs().get(0).get_valueType() 
				+ OPERAND_DELIMITOR + output_index + DATATYPE_PREFIX + get_dataType() + VALUETYPE_PREFIX + this.get_valueType() ;
		return inst;
	}


	public static Transform constructTransformLop(Lops input1, OperationTypes op, DataType dt, ValueType vt) {
		
		for (Lops lop  : input1.getOutputs()) {
			if ( lop.type == Lops.Type.Transform ) {
				return (Transform)lop;
			}
		}
		
		return new Transform(input1, op, dt, vt);
	}

	public static Transform constructTransformLop(Lops input1, OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		
		for (Lops lop  : input1.getOutputs()) {
			if ( lop.type == Lops.Type.Transform ) {
				return (Transform)lop;
			}
		}
		
		return new Transform(input1, op, dt, vt, et);
	}

 
}