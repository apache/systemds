/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.*;


/**
 * Lop to represent an aggregation.
 * It is used in rowsum, colsum, etc. 
 */

public class Aggregate extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/** Aggregate operation types **/
	
	public enum OperationTypes {Sum,Product,Min,Max,Trace,KahanSum,KahanTrace,Mean,MaxIndex};	
	OperationTypes operation;
 
	private boolean isCorrectionUsed = false;
	private CorrectionLocationType correctionLocation = CorrectionLocationType.INVALID;

	/**
	 * @param input - input lop
	 * @param op - operation type
	 */
	public Aggregate(Lop input, Aggregate.OperationTypes op, DataType dt, ValueType vt ) {
		super(Lop.Type.Aggregate, dt, vt);
		init ( input, op, dt, vt, ExecType.MR );
	}
	
	public Aggregate(Lop input, Aggregate.OperationTypes op, DataType dt, ValueType vt, ExecType et ) {
		super(Lop.Type.Aggregate, dt, vt);
		init ( input, op, dt, vt, et );
	}
	
	private void init (Lop input, Aggregate.OperationTypes op, DataType dt, ValueType vt, ExecType et ) {
		operation = op;	
		this.addInput(input);
		input.addOutput(this);
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		if ( et == ExecType.MR ) {
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.DATAGEN);
			lps.addCompatibility(JobType.REBLOCK);
			this.lps.setProperties( inputs, et, ExecLocation.Reduce, breaksAlignment, aligner, definesMRJob );
		}
		else {
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties( inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
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
			throw new UnsupportedOperationException(this.printErrorLocation() + "Instruction is not defined for Aggregate operation: " + operation);
		}
	}
	
	@Override
	public String getInstructions(String input1, String output) throws LopsException {
		String opcode = getOpcode(); 
		
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( opcode );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();
	}
	
	@Override
	public String getInstructions(int input_index, int output_index) throws LopsException
	{
		boolean isCorrectionApplicable = false;
		
		String opcode = getOpcode(); 
		if (operation == OperationTypes.Mean || operation == OperationTypes.KahanSum || operation == OperationTypes.KahanTrace ) 
			isCorrectionApplicable = true;
		
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( opcode );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(0).prepInputOperand(input_index));
		sb.append( OPERAND_DELIMITOR );

		sb.append( this.prepOutputOperand(output_index));
		
		if ( isCorrectionApplicable )
		{
			// add correction information to the instruction
			sb.append( OPERAND_DELIMITOR );
			sb.append( isCorrectionUsed );
			sb.append( OPERAND_DELIMITOR );
			sb.append( correctionLocation );
		}
		
		return sb.toString();
	}

 
 
}
