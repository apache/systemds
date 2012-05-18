package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.functionobjects.COV;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.COVOperator;
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
	
	public AggregateBinaryCPInstruction(Operator op, 
			CPOperand in1, 
			CPOperand in2, 
			CPOperand in3, 
			CPOperand out, 
			String istr ){
		super(op, in1, in2, in3, out, istr);
		cptype = CPINSTRUCTION_TYPE.AggregateBinary;
	}

	public static AggregateBinaryCPInstruction parseInstruction( String str ) 
		throws DMLRuntimeException {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in3 = null;
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String opcode = InstructionUtils.getOpCode(str);

		if ( opcode.equalsIgnoreCase("ba+*")) {
			parseBinaryInstruction(str, in1, in2, out);
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateBinaryOperator aggbin = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
			return new AggregateBinaryCPInstruction(aggbin, in1, in2, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("cov")) {
			COVOperator cov = new COVOperator(COV.getCOMFnObject());
			String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
			if ( parts.length == 4 ) {
				// CP.cov.mVar0.mVar1.mVar2
				parseBinaryInstruction(str, in1, in2, out);
				return new AggregateBinaryCPInstruction(cov, in1, in2, out, str);
			} else if ( parts.length == 5 ) {
				// CP.cov.mVar0.mVar1.mVar2.mVar3
				in3 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
				parseBinaryInstruction(str, in1, in2, in3, out);
				return new AggregateBinaryCPInstruction(cov, in1, in2, in3, out, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of arguments in Instruction: " + str);
			}
		}
		else {
			throw new DMLRuntimeException("AggregateBinaryInstruction.parseInstruction():: Unknown opcode " + opcode);
		}
		
	}
	
	@Override
	public Data processInstruction(ProgramBlock pb) 
		throws DMLRuntimeException, DMLUnsupportedOperationException{
		
		if(pb == null) System.err.println("pb is null");
		
		String opcode = InstructionUtils.getOpCode(instString);
		
		MatrixObject mat1 = pb.getMatrixVariable(input1.get_name());
		MatrixObject mat2 = pb.getMatrixVariable(input2.get_name());
		String output_name = output.get_name(); 
		
		if ( opcode.equalsIgnoreCase("ba+*")) {
			//long st = System.currentTimeMillis();
			//long begin = st;
			AggregateBinaryOperator ab_op = (AggregateBinaryOperator) optr;
	
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
		else {
			COVOperator cov_op = (COVOperator)optr;
			CM_COV_Object covobj = new CM_COV_Object();
			
			if ( input3 == null ) {
				// Unweighted: cov.mvar0.mvar1.out
				covobj = mat1.covOperations(cov_op, mat2); 
			}
			else {
				// Weighted: cov.mvar0.mvar1.weights.out
				MatrixObject wt = pb.getMatrixVariable(input3.get_name());
				covobj = mat1.covOperations(cov_op, mat2, wt); 
			}
			double val = covobj.getRequiredResult(optr);
			DoubleObject ret = new DoubleObject(output_name, val);
			pb.setVariable(output_name, ret);
			return ret;
		}
	}
	
}
