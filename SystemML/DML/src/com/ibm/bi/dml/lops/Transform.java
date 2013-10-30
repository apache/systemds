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
import com.ibm.bi.dml.parser.Expression.*;


/*
 * Lop to perform transpose/vector to diag operations
 * This lop can change the keys and hence break alignment.
 */

public class Transform extends Lop
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum OperationTypes {
		Transpose,
		VectortoDiagMatrix,
		ReshapeMatrix
	};
	
	private OperationTypes operation = null;
	
	/**
	 * Constructor when we have one input.
	 * @param input
	 * @param op
	 */

	public Transform(Lop input, Transform.OperationTypes op, DataType dt, ValueType vt, ExecType et) 
	{
		super(Lop.Type.Transform, dt, vt);		
		init(input, op, dt, vt, et);
	}
	
	public Transform(Lop input, Transform.OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lop.Type.Transform, dt, vt);		
		init(input, op, dt, vt, ExecType.MR);
	}

	private void init (Lop input, Transform.OperationTypes op, DataType dt, ValueType vt, ExecType et) 
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
			lps.addCompatibility(JobType.REBLOCK);
			lps.addCompatibility(JobType.MMCJ);
			lps.addCompatibility(JobType.MMRJ);
			if( op == OperationTypes.ReshapeMatrix )
				//reshape should be executed in map because we have potentially large intermediate data and want to exploit the combiner.
				this.lps.setProperties( inputs, et, ExecLocation.Map, breaksAlignment, aligner, definesMRJob );
			else
				this.lps.setProperties( inputs, et, ExecLocation.MapOrReduce, breaksAlignment, aligner, definesMRJob );
		}
		else {
			// <code>breaksAlignment</code> is not meaningful when <code>Transform</code> executes in CP. 
			breaksAlignment = false;
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties( inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
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
		
		case ReshapeMatrix:
			// Transform a vector into a diagonal matrix
			return "rshape";
				
		default:
			throw new UnsupportedOperationException(this.printErrorLocation() + "Instruction is not defined for Transform operation " + operation);
				
		}
	}
	
	//CP instructions
	
	@Override
	public String getInstructions(String input1, String output) 
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
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
	}

	@Override
	public String getInstructions(String input1, String input2, String input3, String input4, String output) 
		throws LopsException 
	{
		//only used for reshape
		
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( input1 );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(0).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(0).get_valueType() );
		
		//rows, cols, byrow
		String[] inputX = new String[]{input2,input3,input4};
		for( int i=1; i<=(inputX.length); i++ ) {
			Lop ltmp = getInputs().get(i);
			sb.append( OPERAND_DELIMITOR );
			sb.append( inputX[i-1] );
			sb.append( DATATYPE_PREFIX );
			sb.append( ltmp.get_dataType() );
			sb.append( VALUETYPE_PREFIX );
			sb.append( ltmp.get_valueType() );				
		}
		
		//output
		sb.append( OPERAND_DELIMITOR );
		sb.append( output );
		sb.append( DATATYPE_PREFIX );
		sb.append( get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( get_valueType() );
		
		return sb.toString();
	}
	
	//MR instructions

	@Override 
	public String getInstructions(int input_index, int output_index) 
		throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( input_index );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(0).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(0).get_valueType() ); 
		sb.append( OPERAND_DELIMITOR );
		sb.append( output_index );
		sb.append( DATATYPE_PREFIX );
		sb.append( get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( get_valueType() );
		
		return sb.toString();
	}
	
	@Override 
	public String getInstructions(int input_index1, int input_index2, int input_index3, int input_index4, int output_index) 
		throws LopsException
	{
		//only used for reshape
		
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( input_index1 );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(0).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(0).get_valueType() ); 
		
		//rows		
		Lop input2 = getInputs().get(1); 
		String rowsString = input2.getOutputParameters().getLabel();
		if ( (input2.getExecLocation() == ExecLocation.Data &&
				 !((Data)input2).isLiteral()) || !(input2.getExecLocation() == ExecLocation.Data )){
			rowsString = Lop.VARIABLE_NAME_PLACEHOLDER + rowsString + Lop.VARIABLE_NAME_PLACEHOLDER;
		}
		sb.append( OPERAND_DELIMITOR );
		sb.append( rowsString );
		
		//cols
		Lop input3 = getInputs().get(2); 
		String colsString = input3.getOutputParameters().getLabel();
		if ( input3.getExecLocation() == ExecLocation.Data 
				&& !((Data)input3).isLiteral() || !(input3.getExecLocation() == ExecLocation.Data )) {
			colsString = Lop.VARIABLE_NAME_PLACEHOLDER + colsString + Lop.VARIABLE_NAME_PLACEHOLDER;
		}
		sb.append( OPERAND_DELIMITOR );
		sb.append( colsString );
		
		//byrow
		Lop input4 = getInputs().get(3); 
		String byrowString = input4.getOutputParameters().getLabel();
		if ( input4.getExecLocation() == ExecLocation.Data 
				&& !((Data)input4).isLiteral() || !(input4.getExecLocation() == ExecLocation.Data ) ){
			throw new LopsException(this.printErrorLocation() + "Parameter 'byRow' must be a literal for a matrix operation.");
		}
		sb.append( OPERAND_DELIMITOR );
		sb.append( byrowString );
		
		//output
		sb.append( OPERAND_DELIMITOR );
		sb.append( output_index );
		sb.append( DATATYPE_PREFIX );
		sb.append( get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( get_valueType() );
		
		return sb.toString();
	}


	public static Transform constructTransformLop(Lop input1, OperationTypes op, DataType dt, ValueType vt) {
		
		for (Lop lop  : input1.getOutputs()) {
			if ( lop.type == Lop.Type.Transform ) {
				return (Transform)lop;
			}
		}
		Transform retVal = new Transform(input1, op, dt, vt);
		retVal.setAllPositions(input1.getBeginLine(), input1.getBeginColumn(), input1.getEndLine(), input1.getEndColumn());
		return retVal;
	}

	public static Transform constructTransformLop(Lop input1, OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		
		for (Lop lop  : input1.getOutputs()) {
			if ( lop.type == Lop.Type.Transform ) {
				return (Transform)lop;
			}
		}
		Transform retVal = new  Transform(input1, op, dt, vt, et);
		retVal.setAllPositions(input1.getBeginLine(), input1.getBeginColumn(), input1.getEndLine(), input1.getEndColumn());
		return retVal; 
	}

 
}