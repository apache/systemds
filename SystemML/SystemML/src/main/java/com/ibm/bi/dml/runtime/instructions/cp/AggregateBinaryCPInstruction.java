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
import com.ibm.bi.dml.runtime.functionobjects.COV;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.COVOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class AggregateBinaryCPInstruction extends BinaryCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public AggregateBinaryCPInstruction(Operator op, 
										CPOperand in1, 
										CPOperand in2, 
										CPOperand out, 
										String opcode,
										String istr ){
		super(op, in1, in2, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.AggregateBinary;
	}
	
	public AggregateBinaryCPInstruction(Operator op, 
			CPOperand in1, 
			CPOperand in2, 
			CPOperand in3, 
			CPOperand out,
			String opcode,
			String istr ){
		super(op, in1, in2, in3, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.AggregateBinary;
	}

	public static AggregateBinaryCPInstruction parseInstruction( String str ) 
		throws DMLRuntimeException {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in3 = null;
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if ( opcode.equalsIgnoreCase("ba+*")) 
		{
			InstructionUtils.checkNumFields( parts, 4 );
			in1.split(parts[1]);
			in2.split(parts[2]);
			out.split(parts[3]);
			int k = Integer.parseInt(parts[4]);
			
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateBinaryOperator aggbin = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg, k);
			return new AggregateBinaryCPInstruction(aggbin, in1, in2, out, opcode, str);
		} 
		else if ( opcode.equalsIgnoreCase("cov")) {
			COVOperator cov = new COVOperator(COV.getCOMFnObject());
			if ( parts.length == 4 ) {
				// CP.cov.mVar0.mVar1.mVar2
				parseBinaryInstruction(str, in1, in2, out);
				return new AggregateBinaryCPInstruction(cov, in1, in2, out, opcode, str);
			} else if ( parts.length == 5 ) {
				// CP.cov.mVar0.mVar1.mVar2.mVar3
				in3 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
				parseBinaryInstruction(str, in1, in2, in3, out);
				return new AggregateBinaryCPInstruction(cov, in1, in2, in3, out, opcode, str);
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
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{	
		String opcode = getOpcode();
		
		MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());
        MatrixBlock matBlock2 = ec.getMatrixInput(input2.getName());
		String output_name = output.getName(); 
		
		if ( opcode.equalsIgnoreCase("ba+*")) {
			
			AggregateBinaryOperator ab_op = (AggregateBinaryOperator) _optr;
			MatrixBlock soresBlock = (MatrixBlock) (matBlock1.aggregateBinaryOperations(matBlock1, matBlock2, new MatrixBlock(), ab_op));
			
			//release inputs/outputs
			ec.releaseMatrixInput(input1.getName());
			ec.releaseMatrixInput(input2.getName());
			ec.setMatrixOutput(output_name, soresBlock);
			
		} 
		else if ( opcode.equalsIgnoreCase("cov") ) {
			COVOperator cov_op = (COVOperator)_optr;
			CM_COV_Object covobj = new CM_COV_Object();
			
			if ( input3 == null ) 
			{
				// Unweighted: cov.mvar0.mvar1.out
				covobj = matBlock1.covOperations(cov_op, matBlock2);
				
				ec.releaseMatrixInput(input1.getName());
				ec.releaseMatrixInput(input2.getName());
			}
			else 
			{
				// Weighted: cov.mvar0.mvar1.weights.out
		        MatrixBlock wtBlock = ec.getMatrixInput(input3.getName());
				
				covobj = matBlock1.covOperations(cov_op, matBlock2, wtBlock);
				
				ec.releaseMatrixInput(input1.getName());
				ec.releaseMatrixInput(input2.getName());
				ec.releaseMatrixInput(input3.getName());
			}
			double val = covobj.getRequiredResult(_optr);
			DoubleObject ret = new DoubleObject(output_name, val);
			
			ec.setScalarOutput(output_name, ret);
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
		return (AggregateBinaryOperator) _optr;
	}
}
