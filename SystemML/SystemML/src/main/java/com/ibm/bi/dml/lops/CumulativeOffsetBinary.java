/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.Aggregate.OperationTypes;
import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.*;


/**
 * 
 * 
 */
public class CumulativeOffsetBinary extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private OperationTypes _op;
	private double _initValue = 0;
	
	public CumulativeOffsetBinary(Lop data, Lop offsets, DataType dt, ValueType vt, OperationTypes op, ExecType et)
		throws LopsException 
	{
		super(Lop.Type.CumulativeOffsetBinary, dt, vt);
		checkSupportedOperations(op);
		_op = op;
	
		init(data, offsets, dt, vt, et);
	}
	
	public CumulativeOffsetBinary(Lop data, Lop offsets, DataType dt, ValueType vt, double init, OperationTypes op, ExecType et)
		throws LopsException 
	{
		super(Lop.Type.CumulativeOffsetBinary, dt, vt);
		checkSupportedOperations(op);
		_op = op;
		
		//in case of Spark, CumulativeOffset includes CumulativeSplit and hence needs the init value
		_initValue = init;
		
		init(data, offsets, dt, vt, et);
	}
	
	/**
	 * 
	 * @param input
	 * @param dt
	 * @param vt
	 * @param et
	 */
	private void init(Lop input1, Lop input2, DataType dt, ValueType vt, ExecType et) 
	{
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);

		if( et == ExecType.MR )
		{
			//setup MR parameters
			boolean breaksAlignment = true;
			boolean aligner = false;
			boolean definesMRJob = false;
			
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.DATAGEN);
			lps.setProperties(inputs, et, ExecLocation.Reduce, breaksAlignment, aligner, definesMRJob);
		}
		else //Spark/CP
		{
			//setup Spark parameters 
			boolean breaksAlignment = false;
			boolean aligner = false;
			boolean definesMRJob = false;
			lps.addCompatibility(JobType.INVALID);
			lps.setProperties( inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		}
	}

	public String toString() {
		return "CumulativeOffsetBinary";
	}

	/**
	 * 
	 * @param op
	 * @throws LopsException
	 */
	private void checkSupportedOperations(OperationTypes op) 
		throws LopsException
	{
		//sanity check for supported aggregates
		if( !(op == OperationTypes.KahanSum || op == OperationTypes.Product ||
		      op == OperationTypes.Min || op == OperationTypes.Max) )
		{
			throw new LopsException("Unsupported aggregate operation type: "+op);
		}
	}
	
	private String getOpcode() {
		switch( _op ) {
			case KahanSum: 	return "bcumoffk+";
			case Product: 	return "bcumoff*";
			case Min:		return "bcumoffmin";
			case Max: 		return "bcumoffmax";
			default: 		return null;
		}
	}
	
	@Override
	public String getInstructions(int input_index1, int input_index2, int output_index)
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index1) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(input_index2) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output_index) );

		return sb.toString();
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output)
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input1) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(input2) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output) );
		if( getExecType() == ExecType.SPARK ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _initValue );	
		}
		
		return sb.toString();
	}
}
