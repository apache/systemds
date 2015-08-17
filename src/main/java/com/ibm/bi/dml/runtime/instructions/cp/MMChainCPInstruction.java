/**
�* IBM Confidential
�* OCO Source Materials
�* (C) Copyright IBM Corp. 2010, 2015
�* The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
�*/

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.lops.MapMultChain.ChainType;
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
public class MMChainCPInstruction extends UnaryCPInstruction
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private ChainType _type = null;
	private int _numThreads = -1;
	
	public MMChainCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, ChainType type, int k, String opcode, String istr)
	{
		super(op, in1, in2, in3, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.MMChain;
		_type = type;
		_numThreads = k;
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
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in3 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		String opcode = InstructionUtils.getOpCode(str);
		
		//check number of fields (2/3 inputs, output, type)
		InstructionUtils.checkNumFields ( str, 5, 6 );
		
		//parse instruction parts (without exec type)
		String[] parts = InstructionUtils.getInstructionPartsWithValueType( str );		
		in1.split(parts[1]);
		in2.split(parts[2]);
		
		if( parts.length==6 )
		{
			out.split(parts[3]);
			ChainType type = ChainType.valueOf(parts[4]);
			int k = Integer.parseInt(parts[5]);
			
			return new MMChainCPInstruction(null, in1, in2, null, out, type, k, opcode, str);
		}
		else //parts.length==7
		{
			in3.split(parts[3]);
			out.split(parts[4]);
			ChainType type = ChainType.valueOf(parts[5]);
			int k = Integer.parseInt(parts[6]);
			
			return new MMChainCPInstruction(null, in1, in2, in3, out, type, k, opcode, str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		//get inputs
		MatrixBlock X = ec.getMatrixInput(input1.getName());
		MatrixBlock v = ec.getMatrixInput(input2.getName());
		MatrixBlock w = (_type==ChainType.XtwXv) ? ec.getMatrixInput(input3.getName()) : null;

		//execute mmchain operation 
		 MatrixBlock out = (MatrixBlock) X.chainMatrixMultOperations(v, w, new MatrixBlock(), _type, _numThreads);
				
		//set output and release inputs
		ec.setMatrixOutput(output.getName(), out);
		ec.releaseMatrixInput(input1.getName());
		ec.releaseMatrixInput(input2.getName());
		if( w !=null )
			ec.releaseMatrixInput(input3.getName());
	}
	
	public ChainType getMMChainType()
	{
		return _type;
	}
}
