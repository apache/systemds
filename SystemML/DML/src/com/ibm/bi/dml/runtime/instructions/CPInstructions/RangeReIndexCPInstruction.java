package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;

public class RangeReIndexCPInstruction extends UnaryCPInstruction{

	/*
	 * This class implements the matrix indexing functionality inside CP.  
	 * Example instructions: 
	 *     rangeReIndex:mVar1:Var2:Var3:Var4:Var5:mVar6
	 *     input=mVar1, output=mVar6, 
	 *     bounds = (Var2,Var3,Var4,Var5)
	 *     rowindex_lower: Var2, rowindex_upper: Var3 
	 *     colindex_lower: Var4, colindex_upper: Var5
	 *  
	 */
	CPOperand rowLower, rowUpper, colLower, colUpper;
	
	public RangeReIndexCPInstruction(Operator op, CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String istr){
		super(op, in, out, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException {
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		
		if ( parts[0].equalsIgnoreCase("rangeReIndex") ) {
			if ( parts.length == 7 ) {
				// Example: rangeReIndex:mVar1:Var2:Var3:Var4:Var5:mVar6
				CPOperand in, rl, ru, cl, cu, out;
				in = new CPOperand();
				rl = new CPOperand();
				ru = new CPOperand();
				cl = new CPOperand();
				cu = new CPOperand();
				out = new CPOperand();
				in.split(parts[1]);
				rl.split(parts[2]);
				ru.split(parts[3]);
				cl.split(parts[4]);
				cu.split(parts[5]);
				out.split(parts[6]);
				return new RangeReIndexCPInstruction(new SimpleOperator(null), in, rl, ru, cl, cu, out, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		} 
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a RangeReIndexCPInstruction: " + str);
		}
	}
	
	@Override
	public MatrixObject processInstruction(ProgramBlock pb)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		MatrixObject mat = pb.getMatrixVariable(input1.get_name());
		
		long rl = pb.getScalarVariable(rowLower.get_name(), ValueType.INT).getLongValue();
		long ru = pb.getScalarVariable(rowUpper.get_name(), ValueType.INT).getLongValue();
		long cl = pb.getScalarVariable(colLower.get_name(), ValueType.INT).getLongValue();
		long cu = pb.getScalarVariable(colUpper.get_name(), ValueType.INT).getLongValue();
		
		String output_name = output.get_name();
		MatrixObject sores = mat.indexOperations(rl, ru, cl, cu, (MatrixObject)pb.getVariable(output_name));
		
		pb.setVariableAndWriteToHDFS(output_name, sores);
		return sores; 
	}
}
