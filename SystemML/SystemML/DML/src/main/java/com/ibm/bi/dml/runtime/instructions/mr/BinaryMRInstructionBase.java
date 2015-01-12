/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.mr;

import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public abstract class BinaryMRInstructionBase extends MRInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public byte input1, input2;
	
	public BinaryMRInstructionBase(Operator op, byte in1, byte in2, byte out)
	{
		super(op, out);
		input1=in1;
		input2=in2;
	}
	
	@Override
	public byte[] getInputIndexes() {
		return new byte[]{input1, input2};
	}

	@Override
	public byte[] getAllIndexes() {
		return new byte[]{input1, input2, output};
	}

}
