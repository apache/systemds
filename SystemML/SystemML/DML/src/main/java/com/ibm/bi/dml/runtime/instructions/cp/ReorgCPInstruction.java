/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.DiagIndex;
import com.ibm.bi.dml.runtime.functionobjects.SortIndex;
import com.ibm.bi.dml.runtime.functionobjects.SwapIndex;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;


public class ReorgCPInstruction extends UnaryCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//sort-specific attributes (to enable variable attributes)
 	private CPOperand _col = null;
 	private CPOperand _desc = null;
 	private CPOperand _ixret = null;
 	
 	/**
 	 * for opcodes r' and rdiag
 	 * 
 	 * @param op
 	 * @param in
 	 * @param out
 	 * @param opcode
 	 * @param istr
 	 */
	public ReorgCPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr){
		super(op, in, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.Reorg;
	}
	
	/**
	 * for opcode rsort
	 * 
	 * @param op
	 * @param in
	 * @param col
	 * @param desc
	 * @param ixret
	 * @param out
	 * @param opcode
	 * @param istr
	 */
	public ReorgCPInstruction(Operator op, CPOperand in, CPOperand col, CPOperand desc, CPOperand ixret, CPOperand out, String opcode, String istr){
		this(op, in, out, opcode, istr);
		_col = col;
		_desc = desc;
		_ixret = ixret;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = InstructionUtils.getOpCode(str);
		
		if ( opcode.equalsIgnoreCase("r'") ) {
			parseUnaryInstruction(str, in, out); //max 2 operands
			return new ReorgCPInstruction(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out, opcode, str);
		} 
		else if ( opcode.equalsIgnoreCase("rdiag") ) {
			parseUnaryInstruction(str, in, out); //max 2 operands
			return new ReorgCPInstruction(new ReorgOperator(DiagIndex.getDiagIndexFnObject()), in, out, opcode, str);
		} 
		else if ( opcode.equalsIgnoreCase("rsort") ) {
			InstructionUtils.checkNumFields(str, 5);
			String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
			in.split(parts[1]);
			out.split(parts[5]);
			CPOperand col = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			CPOperand desc = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			CPOperand ixret = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			col.split(parts[2]);
			desc.split(parts[3]);
			ixret.split(parts[4]);
			
			return new ReorgCPInstruction(new ReorgOperator(SortIndex.getSortIndexFnObject(1,false,false)), 
					                      in, col, desc, ixret, out, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a ReorgInstruction: " + str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		//acquire inputs
		MatrixBlock matBlock = ec.getMatrixInput(input1.getName());		
		ReorgOperator r_op = (ReorgOperator) _optr;
		if( r_op.fn instanceof SortIndex ) {
			//additional attributes for sort
			int col = (int)ec.getScalarInput(_col.getName(), _col.getValueType(), _col.isLiteral()).getLongValue();
			boolean desc = ec.getScalarInput(_desc.getName(), _desc.getValueType(), _desc.isLiteral()).getBooleanValue();
			boolean ixret = ec.getScalarInput(_ixret.getName(), _ixret.getValueType(), _ixret.isLiteral()).getBooleanValue();
			r_op.fn = SortIndex.getSortIndexFnObject(col, desc, ixret);
		}
		
		//execute operation
		MatrixBlock soresBlock = (MatrixBlock) (matBlock.reorgOperations(r_op, new MatrixBlock(), 0, 0, 0));
        
		//release inputs/outputs
		ec.releaseMatrixInput(input1.getName());
		ec.setMatrixOutput(output.getName(), soresBlock);
	}
	
}
