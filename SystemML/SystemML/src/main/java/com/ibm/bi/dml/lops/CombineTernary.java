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

public class CombineTernary extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum OperationTypes {
		PreCovWeighted, PreGroupedAggWeighted
	}; // PreCovUnweighted,PreGroupedAggWeighted will be CombineBinary

	OperationTypes operation;

	public CombineTernary( OperationTypes op, Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt) {
		super(Lop.Type.CombineTernary, dt, vt);
		operation = op;
		this.addInput(input1);
		this.addInput(input2);
		this.addInput(input3);
		input1.addOutput(this);
		input2.addOutput(this);
		input3.addOutput(this);

		/*
		 * This lop can ONLY be executed as a STANDALONE job
		 */
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = true;
		this.lps.addCompatibility(JobType.COMBINE);
		this.lps.setProperties(inputs, ExecType.MR, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
		this.lps.setProducesIntermediateOutput(true);
	}

	public String toString() {
		return "combineternary";
	}

	@Override
	public String getInstructions(int input_index1, int input_index2,
			int input_index3, int output_index) 
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( "combinetertiary" );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(0).prepInputOperand(input_index1));
		sb.append( OPERAND_DELIMITOR );

		sb.append( getInputs().get(1).prepInputOperand(input_index2));
		sb.append( OPERAND_DELIMITOR );

		sb.append( getInputs().get(2).prepInputOperand(input_index3));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( this.prepInputOperand(output_index));

		return sb.toString();
	}

	public OperationTypes getOperation() {
		return operation;
	}

	public static CombineTernary constructCombineLop( OperationTypes op, Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt) {

		HashSet<Lop> set1 = new HashSet<Lop>();
		set1.addAll(input1.getOutputs());

		// find intersection of input1.getOutputs() and input2.getOutputs()
		set1.retainAll(input2.getOutputs());

		// find intersection of the above result and input3.getOutputs()
		set1.retainAll(input3.getOutputs());
		
		for (Lop lop : set1) {
			if (lop.type == Lop.Type.CombineTernary) {
				CombineTernary combine = (CombineTernary) lop;
				if (combine.operation == op)
					return (CombineTernary) lop;
			}
		}

		CombineTernary comn = new CombineTernary(op, input1, input2, input3, dt, vt);
		comn.setAllPositions(input1.getBeginLine(), input1.getBeginColumn(), input1.getEndLine(), input1.getEndColumn());
		return comn;
	}

}
