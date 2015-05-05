/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.lops.WeightedSquaredLoss.WeightsType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;

/**
 * 
 * 
 */
public class QuaternaryCPInstruction extends ComputationCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private WeightsType wtype = null;
	private CPOperand input4 = null;
 	
	
	public QuaternaryCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4, CPOperand out, 
							       WeightsType wt, String opcode, String istr )
	{
		super(op, in1, in2, in3, out, opcode, istr);
		
		input4 = in4;
		wtype = wt;
	}

	public static QuaternaryCPInstruction parseInstruction(String inst) 
		throws DMLRuntimeException
	{	
		InstructionUtils.checkNumFields ( inst, 6 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(inst);
		String opcode = parts[0];
		
		//handle opcode
		if ( !opcode.equalsIgnoreCase("wsloss") ) {
			throw new DMLRuntimeException("Unexpected opcode in QuaternaryCPInstruction: " + inst);
		}
		
		//handle operands
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		CPOperand in4 = new CPOperand(parts[4]);
		CPOperand out = new CPOperand(parts[5]);
		
		WeightsType wtype = WeightsType.valueOf(parts[6]);
		
		return new QuaternaryCPInstruction(new SimpleOperator(null), in1, in2, in3, in4, out, wtype, opcode, inst);
	}

	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());
		MatrixBlock matBlock2 = ec.getMatrixInput(input2.getName());
		MatrixBlock matBlock3 = ec.getMatrixInput(input3.getName());
		MatrixBlock matBlock4 = null;
		if( wtype != WeightsType.NONE )
			matBlock4 = ec.getMatrixInput(input4.getName());
		
		//core execute
		MatrixValue out = matBlock1.quaternaryOperations(_optr, matBlock2, matBlock3, matBlock4, new MatrixBlock(), wtype);
		DoubleObject ret = new DoubleObject(out.getValue(0, 0));
		
		//release inputs and output
		ec.releaseMatrixInput(input1.getName());
		ec.releaseMatrixInput(input2.getName());
		ec.releaseMatrixInput(input3.getName());
		if( wtype != WeightsType.NONE )
			ec.releaseMatrixInput(input4.getName());
		ec.setVariable(output.getName(), ret);
	}	
}
