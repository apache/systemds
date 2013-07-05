package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.lops.MMTSJ.MMTSJType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.SymbolTable;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;

/**
 * 
 * 
 */
public class MMTSJCPInstruction extends UnaryCPInstruction
{	
	private MMTSJType _type = null;
	
	public MMTSJCPInstruction(Operator op, CPOperand in1, MMTSJType type, CPOperand out, String istr)
	{
		super(op, in1, out, istr);
		cptype = CPINSTRUCTION_TYPE.MMTSJ;
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
			throw new DMLRuntimeException("Unknown opcode while parsing an MMTIJCPInstruction: " + str);
		else
			return new MMTSJCPInstruction(new Operator(true), in1, titype, out, str);
	}
	
	@Override
	public void processInstruction(SymbolTable symb)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		//get inputs
		MatrixBlock matBlock1 = symb.getMatrixInput(input1.get_name());

		//execute operations 
		MatrixBlock ret = (MatrixBlock) matBlock1.transposeSelfMatrixMult(new MatrixBlock(), _type );
		
		//set output and release inputs
		String output_name = output.get_name();
		symb.setMatrixOutput(output_name, ret);
		symb.releaseMatrixInput(input1.get_name());
	}
	
	public MMTSJType getMMTSJType()
	{
		return _type;
	}
}
