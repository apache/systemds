package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.SymbolTable;
import com.ibm.bi.dml.runtime.functionobjects.COV;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
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
	public void processInstruction(SymbolTable symb) 
		throws DMLRuntimeException, DMLUnsupportedOperationException{
		
		if(symb == null) System.err.println("Symbol table is null");
		
		String opcode = InstructionUtils.getOpCode(instString);
		
		long begin, st, tread, tcompute, twrite, ttotal;
		
		begin = System.currentTimeMillis();
        MatrixBlock matBlock1 = symb.getMatrixInput(input1.get_name());
        MatrixBlock matBlock2 = symb.getMatrixInput(input2.get_name());
		tread = System.currentTimeMillis() - begin;
		
		String output_name = output.get_name(); 
		
		if ( opcode.equalsIgnoreCase("ba+*")) {
			
			st = System.currentTimeMillis();
			AggregateBinaryOperator ab_op = (AggregateBinaryOperator) optr;
			MatrixBlock soresBlock = (MatrixBlock) (matBlock1.aggregateBinaryOperations(matBlock1, matBlock2, new MatrixBlock(), ab_op));
			tcompute = System.currentTimeMillis() - st;

			//remove redundant representation (see examSparsity in matrixmult)
			soresBlock.cleanUp(); 

			//release inputs/outputs
			st = System.currentTimeMillis();
			symb.releaseMatrixInput(input1.get_name());
			symb.releaseMatrixInput(input2.get_name());
			symb.setMatrixOutput(output_name, soresBlock);
			twrite = System.currentTimeMillis()-st;
			
			ttotal = System.currentTimeMillis()-begin;
			LOG.trace("CPInst " + this.toString() + "\t" + tread + "\t" + tcompute + "\t" + twrite + "\t" + ttotal);			
		} 
		else if ( opcode.equalsIgnoreCase("cov") ) {
			COVOperator cov_op = (COVOperator)optr;
			CM_COV_Object covobj = new CM_COV_Object();
			
			if ( input3 == null ) 
			{
				// Unweighted: cov.mvar0.mvar1.out
				covobj = matBlock1.covOperations(cov_op, matBlock2);
				
				matBlock1 = matBlock2 = null;
				symb.releaseMatrixInput(input1.get_name());
				symb.releaseMatrixInput(input2.get_name());
			}
			else 
			{
				// Weighted: cov.mvar0.mvar1.weights.out
		        MatrixBlock wtBlock = symb.getMatrixInput(input3.get_name());
				
				covobj = matBlock1.covOperations(cov_op, matBlock2, wtBlock);
				
				matBlock1 = matBlock2 = wtBlock = null;
				symb.releaseMatrixInput(input1.get_name());
				symb.releaseMatrixInput(input2.get_name());
				symb.releaseMatrixInput(input3.get_name());
			}
			double val = covobj.getRequiredResult(optr);
			DoubleObject ret = new DoubleObject(output_name, val);
			
			symb.setScalarOutput(output_name, ret);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode in Instruction: " + toString());
		}
	}

	/**
	 * NOTE: This method is only used for experiments.
	 * 
	 * @return
	 */
	@Deprecated
	public AggregateBinaryOperator getAggregateOperator()
	{
		return (AggregateBinaryOperator) optr;
	}
}
