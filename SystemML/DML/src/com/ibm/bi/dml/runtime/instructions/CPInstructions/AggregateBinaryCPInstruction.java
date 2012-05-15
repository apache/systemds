package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class AggregateBinaryCPInstruction extends BinaryCPInstruction{
	public AggregateBinaryCPInstruction(Operator op, 
										CPOperand in1, 
										CPOperand in2, 
										CPOperand out, 
										String istr ){
		super(op, in1, in2, out, istr);
		cptype = CPINSTRUCTION_TYPE.AggregateBinary;
	}
	
	public static AggregateBinaryCPInstruction parseInstruction( String str ) 
		throws DMLRuntimeException {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = parseBinaryInstruction(str, in1, in2, out);

		if ( opcode.equalsIgnoreCase("ba+*")) {
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateBinaryOperator aggbin = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
			return new AggregateBinaryCPInstruction(aggbin, in1, in2, out, str);
		} 
		else {
			throw new DMLRuntimeException("AggregateBinaryInstruction.parseInstruction():: Unknown opcode " + opcode);
		}
		
	}
	
	@Override
	public MatrixObject processInstruction(ProgramBlock pb) 
		throws DMLRuntimeException, DMLUnsupportedOperationException{
		
		//long st = System.currentTimeMillis();
		//long begin = st;
		if(pb == null) System.err.println("pb is null");
		
		//System.out.println("MatMult in CP ...");
		
		MatrixObject mat1 = pb.getMatrixVariable(input1.get_name());
		MatrixObject mat2 = pb.getMatrixVariable(input2.get_name());
		AggregateBinaryOperator ab_op = (AggregateBinaryOperator) optr;
		String output_name = output.get_name(); 

		//System.out.println("    setup inputs: " + (System.currentTimeMillis()-st) + " msec.");
		//st = System.currentTimeMillis();
		
		MatrixObject res = mat1.aggregateBinaryOperations(ab_op, mat2, (MatrixObject)pb.getVariable(output_name));
		
		//System.out.println("    compute result: " + (System.currentTimeMillis()-st) + " msec.");
		//st = System.currentTimeMillis();
		
		pb.setVariableAndWriteToHDFS(output_name, res);
		//System.out.println("    write to HDFS: " + (System.currentTimeMillis()-st) + " msec.");
		//System.out.println("    Total: " + (System.currentTimeMillis()-begin) + " msec.");
		
		return res; 
	}
	
}
