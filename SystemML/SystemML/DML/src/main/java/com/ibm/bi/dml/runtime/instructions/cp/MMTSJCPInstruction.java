/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.lops.MMTSJ.MMTSJType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

/**
 * 
 * 
 */
public class MMTSJCPInstruction extends UnaryCPInstruction
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private MMTSJType _type = null;
	
	public MMTSJCPInstruction(Operator op, CPOperand in1, MMTSJType type, CPOperand out, String opcode, String istr)
	{
		super(op, in1, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.MMTSJ;
		_type = type;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		in1.split(parts[1]);
		out.split(parts[2]);
		MMTSJType titype = MMTSJType.valueOf(parts[3]);
		 
		if(!opcode.equalsIgnoreCase("tsmm"))
			throw new DMLRuntimeException("Unknown opcode while parsing an MMTSJCPInstruction: " + str);
		else
			return new MMTSJCPInstruction(new Operator(true), in1, titype, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		//get inputs
		MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());

		//execute operations 
		MatrixBlock ret = (MatrixBlock) matBlock1.transposeSelfMatrixMultOperations(new MatrixBlock(), _type );
		
		//set output and release inputs
		ec.setMatrixOutput(output.getName(), ret);
		ec.releaseMatrixInput(input1.getName());
	}
	
	public MMTSJType getMMTSJType()
	{
		return _type;
	}
}
