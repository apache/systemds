/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import java.util.HashSet;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


/**
 * Lop to represent an combine operation -- used ONLY in the context of sort.
 */

public class CombineBinary extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum OperationTypes {PreSort, PreCentralMoment, PreCovUnweighted, PreGroupedAggUnweighted}; // (PreCovWeighted,PreGroupedAggWeighted) will be CombineTertiary	
	OperationTypes operation;

	/**
	 * @param input - input lop
	 * @param op - operation type
	 */
	
	public CombineBinary(OperationTypes op, Lop input1, Lop input2, DataType dt, ValueType vt) 
	{
		super(Lop.Type.CombineBinary, dt, vt);	
		operation = op;
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		
		/*
		 *  This lop can ONLY be executed as a STANDALONE job
		 */
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = true;
		lps.addCompatibility(JobType.COMBINE);
		this.lps.setProperties( inputs, ExecType.MR, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob );
	}
	
	public String toString()
	{
		return "combinebinary";		
	}

	@Override
	public String getInstructions(int input_index1, int input_index2, int output_index) 
		throws LopsException
	{
		// Determine whether or not the second input denotes weights vector.
		// CombineBinary can be used to combine (data,weights) vectors or (data1,data2) vectors  
		boolean isSecondInputIsWeight = true;
		if ( operation == OperationTypes.PreCovUnweighted || operation == OperationTypes.PreGroupedAggUnweighted ) {
			isSecondInputIsWeight = false;
		}
		
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( "combinebinary" );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( prepOperand(String.valueOf(isSecondInputIsWeight), DataType.SCALAR, ValueType.BOOLEAN) );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(0).prepInputOperand(input_index1));
		sb.append( OPERAND_DELIMITOR );

		sb.append( getInputs().get(1).prepInputOperand(input_index2));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( this.prepOutputOperand(output_index));
		
		return sb.toString();
	}

	public OperationTypes getOperation() { 
		return operation;
	}
	
	public static CombineBinary constructCombineLop(OperationTypes op, Lop input1, 
			Lop input2, DataType dt, ValueType vt) {
		
		HashSet<Lop> set1 = new HashSet<Lop>();
		set1.addAll(input1.getOutputs());
		
		// find intersection of input1.getOutputs() and input2.getOutputs();
		set1.retainAll(input2.getOutputs());
		
		for (Lop lop  : set1) {
			if ( lop.type == Lop.Type.CombineBinary ) {
				CombineBinary combine = (CombineBinary)lop;
				if ( combine.operation == op)
					return (CombineBinary)lop;
			}
		}
		
		CombineBinary comn = new CombineBinary(op, input1, input2, dt, vt);
		comn.setAllPositions(input1.getBeginLine(), input1.getBeginColumn(), input1.getEndLine(), input1.getEndColumn());
		return comn;
	}
 
}
