package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.functionobjects.MaxIndex;
import com.ibm.bi.dml.runtime.functionobjects.SwapIndex;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class ReorgCPInstruction extends UnaryCPInstruction{
	public ReorgCPInstruction(Operator op, CPOperand in, CPOperand out, String istr){
		super(op, in, out, istr);
		cptype = CPINSTRUCTION_TYPE.Reorg;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = parseUnaryInstruction(str, in, out);
		
		if ( opcode.equalsIgnoreCase("r'") ) {
			return new ReorgCPInstruction(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("rdiagV2M") ) {
			return new ReorgCPInstruction(new ReorgOperator(MaxIndex.getMaxIndexFnObject()), in, out, str);
		} 
		
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a ReorgInstruction: " + str);
		}
	}
	
	@Override
	public void processInstruction(ProgramBlock pb)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		long begin, st, tread, tcompute, twrite, ttotal;
		
		begin = System.currentTimeMillis();
		MatrixBlock matBlock = (MatrixBlock) pb.getMatrixInput(input1.get_name());
		tread = System.currentTimeMillis() - begin;
		
		st = System.currentTimeMillis();
		ReorgOperator r_op = (ReorgOperator) optr;
		String output_name = output.get_name();
		
		MatrixBlock soresBlock = (MatrixBlock) (matBlock.reorgOperations (r_op, new MatrixBlock(), 0, 0, 0));
        
		tcompute = System.currentTimeMillis() - st;
		
		st = System.currentTimeMillis();
		matBlock = null;
		pb.releaseMatrixInput(input1.get_name());
		pb.setMatrixOutput(output_name, soresBlock);
		soresBlock = null;
		
		twrite = System.currentTimeMillis() - st;
		ttotal = System.currentTimeMillis()-begin;
		
		if( DMLScript.DEBUG )
			System.out.println("CPInst " + this.toString() + "\t" + tread + "\t" + tcompute + "\t" + twrite + "\t" + ttotal);
	}
	
}
