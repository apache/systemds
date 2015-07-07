/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.lops.WeightedSigmoid.WSigmoidType;
import com.ibm.bi.dml.lops.WeightedSquaredLoss.WeightsType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.QuaternaryOperator;

/**
 * 
 * 
 */
public class QuaternaryCPInstruction extends ComputationCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private CPOperand input4 = null;
 	private int _numThreads = -1;
	
	public QuaternaryCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4, CPOperand out, 
							       int k, String opcode, String istr )
	{
		super(op, in1, in2, in3, out, opcode, istr);
		
		input4 = in4;
		_numThreads = k;
	}

	/**
	 * 
	 * @param inst
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static QuaternaryCPInstruction parseInstruction(String inst) 
		throws DMLRuntimeException
	{	
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(inst);
		String opcode = parts[0];
		
		if( opcode.equalsIgnoreCase("wsloss") ) 
		{
			InstructionUtils.checkNumFields ( inst, 7 );
			
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand in4 = new CPOperand(parts[4]);
			CPOperand out = new CPOperand(parts[5]);
			
			WeightsType wtype = WeightsType.valueOf(parts[6]);
			int k = Integer.parseInt(parts[7]);
			
			return new QuaternaryCPInstruction(new QuaternaryOperator(wtype), in1, in2, in3, in4, out, k, opcode, inst);	
		}
		else if( opcode.equalsIgnoreCase("wsigmoid") )
		{
			InstructionUtils.checkNumFields ( inst, 6 );
			
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[4]);
			
			WSigmoidType wtype = WSigmoidType.valueOf(parts[5]);
			int k = Integer.parseInt(parts[6]);
			
			return new QuaternaryCPInstruction(new QuaternaryOperator(wtype), in1, in2, in3, null, out, k, opcode, inst);
		}
		else {
			throw new DMLRuntimeException("Unexpected opcode in QuaternaryCPInstruction: " + inst);
		}
	}

	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		QuaternaryOperator qop = (QuaternaryOperator) _optr;
		
		MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());
		MatrixBlock matBlock2 = ec.getMatrixInput(input2.getName());
		MatrixBlock matBlock3 = ec.getMatrixInput(input3.getName());
		MatrixBlock matBlock4 = null;
		if( qop.wtype1 != null && qop.wtype1 != WeightsType.NONE )
			matBlock4 = ec.getMatrixInput(input4.getName());
		
		//core execute
		MatrixValue out = matBlock1.quaternaryOperations(qop, matBlock2, matBlock3, matBlock4, new MatrixBlock(), _numThreads);
		
		//release inputs and output
		ec.releaseMatrixInput(input1.getName());
		ec.releaseMatrixInput(input2.getName());
		ec.releaseMatrixInput(input3.getName());
		if( qop.wtype1 != null ) { //wsloss
			if( qop.wtype1 != WeightsType.NONE )
				ec.releaseMatrixInput(input4.getName());
			ec.setVariable(output.getName(), new DoubleObject(out.getValue(0, 0)));
		}
		else if( qop.wtype2 != null ) { //wsigmoid
			ec.setMatrixOutput(output.getName(), (MatrixBlock)out);
		}
	}	
}
