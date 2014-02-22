/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class ComputationCPInstruction extends CPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	

	public CPOperand output;
	public CPOperand input1, input2, input3;
	
	public ComputationCPInstruction ( Operator op, CPOperand in1, CPOperand in2, CPOperand out ) {
		super(op);
		input1 = in1;
		input2 = in2;
		input3 = null;
		output = out;
	}

	public ComputationCPInstruction ( Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out ) {
		super(op);
		input1 = in1;
		input2 = in2;
		input3 = in3;
		output = out;
	}

	public String getOutputVariableName() {
		return output.get_name();
	}
}
