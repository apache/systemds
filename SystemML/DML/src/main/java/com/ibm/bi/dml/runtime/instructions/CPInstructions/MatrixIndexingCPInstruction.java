/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RangeBasedReIndexInstruction.IndexRange;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;

public class MatrixIndexingCPInstruction extends UnaryCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/*
	 * This class implements the matrix indexing functionality inside CP.  
	 * Example instructions: 
	 *     rangeReIndex:mVar1:Var2:Var3:Var4:Var5:mVar6
	 *         input=mVar1, output=mVar6, 
	 *         bounds = (Var2,Var3,Var4,Var5)
	 *         rowindex_lower: Var2, rowindex_upper: Var3 
	 *         colindex_lower: Var4, colindex_upper: Var5
	 *     leftIndex:mVar1:mVar2:Var3:Var4:Var5:Var6:mVar7
	 *         triggered by "mVar1[Var3:Var4, Var5:Var6] = mVar2"
	 *         the result is stored in mVar7
	 *  
	 */
	protected CPOperand rowLower, rowUpper, colLower, colUpper;
	
	public MatrixIndexingCPInstruction(Operator op, CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String istr){
		super(op, in, out, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
	}
	
	public MatrixIndexingCPInstruction(Operator op, CPOperand lhsInput, CPOperand rhsInput, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String istr){
		super(op, lhsInput, rhsInput, out, istr);
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
				return new MatrixIndexingCPInstruction(new SimpleOperator(null), in, rl, ru, cl, cu, out, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		} 
		else if ( parts[0].equalsIgnoreCase("leftIndex")) {
			if ( parts.length == 8 ) {
				// Example: leftIndex:mVar1:mvar2:Var3:Var4:Var5:Var6:mVar7
				CPOperand lhsInput, rhsInput, rl, ru, cl, cu, out;
				lhsInput = new CPOperand();
				rhsInput = new CPOperand();
				rl = new CPOperand();
				ru = new CPOperand();
				cl = new CPOperand();
				cu = new CPOperand();
				out = new CPOperand();
				lhsInput.split(parts[1]);
				rhsInput.split(parts[2]);
				rl.split(parts[3]);
				ru.split(parts[4]);
				cl.split(parts[5]);
				cu.split(parts[6]);
				out.split(parts[7]);
				return new MatrixIndexingCPInstruction(new SimpleOperator(null), lhsInput, rhsInput, rl, ru, cl, cu, out, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a MatrixIndexingCPInstruction: " + str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		String opcode = InstructionUtils.getOpCode(this.instString);
		
		long rl = ec.getScalarInput(rowLower.get_name(), rowLower.get_valueType(), rowLower.isLiteral()).getLongValue();
		long ru = ec.getScalarInput(rowUpper.get_name(), rowUpper.get_valueType(), rowUpper.isLiteral()).getLongValue();
		long cl = ec.getScalarInput(colLower.get_name(), colLower.get_valueType(), colLower.isLiteral()).getLongValue();
		long cu = ec.getScalarInput(colUpper.get_name(), colUpper.get_valueType(), colUpper.isLiteral()).getLongValue();
		
		MatrixObject mo = (MatrixObject)ec.getVariable(input1.get_name());
		MatrixBlock resultBlock = null;
		
		if( mo.isPartitioned() && opcode.equalsIgnoreCase("rangeReIndex") ) //MB: it will always be rangeReIndex!
		{
			resultBlock = mo.readMatrixPartition( new IndexRange(rl,ru,cl,cu) );
		}
		else
		{
			MatrixBlock matBlock = ec.getMatrixInput(input1.get_name());
			
			if ( opcode.equalsIgnoreCase("rangeReIndex"))
			{
				resultBlock = (MatrixBlock) matBlock.sliceOperations(rl, ru, cl, cu, new MatrixBlock());
			}
			else if ( opcode.equalsIgnoreCase("leftIndex"))
			{
				if(input2.get_dataType() == DataType.MATRIX) //MATRIX<-MATRIX
				{
					MatrixBlock rhsMatBlock = ec.getMatrixInput(input2.get_name());
					resultBlock = (MatrixBlock) matBlock.leftIndexingOperations(rhsMatBlock, rl, ru, cl, cu, new MatrixBlock());
					ec.releaseMatrixInput(input2.get_name());
				}
				else //MATRIX<-SCALAR 
				{
					if(!(rl==ru && cl==cu))
						throw new DMLRuntimeException("Invalid index range of scalar leftindexing: ["+rl+":"+ru+","+cl+":"+cu+"]." );
					ScalarObject scalar = ec.getScalarInput(input2.get_name(), ValueType.DOUBLE, input2.isLiteral());
					resultBlock = (MatrixBlock) matBlock.leftIndexingOperations(scalar, rl, cl, new MatrixBlock());
				}
			}
			else
				throw new DMLRuntimeException("Invalid opcode (" + opcode +") encountered in MatrixIndexingCPInstruction.");
			
			ec.releaseMatrixInput(input1.get_name());
		}
		
		ec.setMatrixOutput(output.get_name(), resultBlock);
	}
}
