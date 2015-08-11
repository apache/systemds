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
 * TODO Additional compiler enhancements:
 * 1) Partial Shuffle Elimination - Any full or aligned blocks could be directly output from the mappers
 *    to the result index. We only need to shuffle, sort and aggregate partial blocks. However, this requires
 *    piggybacking changes, i.e., (1) operations with multiple result indexes, and (2) multiple operations 
 *    with the same result index. 
 * 2) Group Elimination for Append Chains - If we have chains of rappend each intermediate is shuffled and 
 *    aggregated. This is unnecessary if all offsets are known in advance. We could directly pack all rappends
 *    in one GMR map-phase followed by one group and subsequent aggregate. However, this requires an n-ary 
 *    rappend or group (with multiple inputs).
 * 
 */
public class AppendG extends Lop
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String OPCODE = "gappend";
	
	public AppendG(Lop input1, Lop input2, Lop input3, Lop input4, DataType dt, ValueType vt, ExecType et) 
	{
		super(Lop.Type.Append, dt, vt);
		init(input1, input2, input3, input4, dt, vt, et);
	}
	
	public void init(Lop input1, Lop input2, Lop input3, Lop input4, DataType dt, ValueType vt, ExecType et) 
	{
		addInput(input1);
		input1.addOutput(this);

		addInput(input2);
		input2.addOutput(this);
		
		addInput(input3);
		input3.addOutput(this);
		
		addInput(input4);
		input4.addOutput(this);
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		if( et == ExecType.MR )
		{
			lps.addCompatibility(JobType.GMR);
			lps.setProperties( inputs, ExecType.MR, ExecLocation.Map, breaksAlignment, aligner, definesMRJob );
		}
		else //SP
		{
			lps.addCompatibility(JobType.INVALID);
			lps.setProperties( inputs, ExecType.SPARK, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		}
	}
	
	@Override
	public String toString() {
		return " AppendG: ";
	}

	//called when append executes in MR
	public String getInstructions(int input_index1, int input_index2, int input_index3, int input_index4, int output_index) 
		throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( OPCODE );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index1+""));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(input_index2+""));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(2).prepScalarInputOperand(getExecType()));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(3).prepScalarInputOperand(getExecType()));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output_index+"") );
		
		return sb.toString();	
	}
	
	//called when append executes in SP
	public String getInstructions(String input_index1, String input_index2, String input_index3, String input_index4, String output_index) 
		throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( OPCODE );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index1+""));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(input_index2+""));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(2).prepScalarInputOperand(getExecType()));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(3).prepScalarInputOperand(getExecType()));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output_index+"") );
		
		return sb.toString();
	}
}
