/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import java.util.HashMap;

import com.ibm.bi.dml.lops.Tertiary;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;
import com.ibm.bi.dml.runtime.util.DataConverter;


public class TertiaryCPInstruction extends ComputationCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public TertiaryCPInstruction(Operator op, 
							 CPOperand in1, 
							 CPOperand in2, 
							 CPOperand in3, 
							 CPOperand out, 
						     String istr ){
		super(op, in1, in2, in3, out);
		instString = istr;
	}

	public static TertiaryCPInstruction parseInstruction(String inst) throws DMLRuntimeException{
		
		InstructionUtils.checkNumFields ( inst, 4 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(inst);
		String opcode = parts[0];
		
		if ( !opcode.equalsIgnoreCase("ctable") ) {
			throw new DMLRuntimeException("Unexpected opcode in TertiaryCPInstruction: " + inst);
		}
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		CPOperand out = new CPOperand(parts[4]);
		
		// ctable does not require any operator, so we simply pass-in a dummy operator with null functionobject
		return new TertiaryCPInstruction(new SimpleOperator(null), in1, in2, in3, out, inst);
	}
	
/*	static TertiaryOperator getTertiaryOperator(String opcode) throws DMLRuntimeException{
		throw new DMLRuntimeException("Unknown tertiary opcode " + opcode);
	}	
*/	

	private Tertiary.OperationTypes findCtableOperation() {
		DataType dt1 = input1.get_dataType();
		DataType dt2 = input2.get_dataType();
		DataType dt3 = input3.get_dataType();
		return Tertiary.findCtableOperationByInputDataTypes(dt1, dt2, dt3);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException {
		
		MatrixBlock matBlock1 = (MatrixBlock) ec.getMatrixInput(input1.get_name());
		MatrixBlock matBlock2=null, wtBlock=null;
		double cst1, cst2;
		
		HashMap<MatrixIndexes,Double> ctableMap = new HashMap<MatrixIndexes,Double>();
		MatrixBlock resultBlock = null;
		Tertiary.OperationTypes ctableOp = findCtableOperation();
		
		switch(ctableOp) {
		case CTABLE_TRANSFORM:
			// F=ctable(A,B,W)
			matBlock2 = (MatrixBlock) ec.getMatrixInput(input2.get_name());
			wtBlock = (MatrixBlock) ec.getMatrixInput(input3.get_name());
			matBlock1.tertiaryOperations((SimpleOperator)optr, matBlock2, wtBlock, ctableMap);
			break;
		case CTABLE_TRANSFORM_SCALAR_WEIGHT:
			// F = ctable(A,B) or F = ctable(A,B,1)
			matBlock2 = (MatrixBlock) ec.getMatrixInput(input2.get_name());
			cst1 = ec.getScalarInput(input3.get_name(), input3.get_valueType(), input3.isLiteral()).getDoubleValue();
			matBlock1.tertiaryOperations((SimpleOperator)optr, matBlock2, cst1, ctableMap);
			break;
		case CTABLE_TRANSFORM_HISTOGRAM:
			// F=ctable(A,1) or F = ctable(A,1,1)
			cst1 = ec.getScalarInput(input2.get_name(), input2.get_valueType(), input2.isLiteral()).getDoubleValue();
			cst2 = ec.getScalarInput(input3.get_name(), input3.get_valueType(), input3.isLiteral()).getDoubleValue();
			matBlock1.tertiaryOperations((SimpleOperator)optr, cst1, cst2, ctableMap);
			break;
		case CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM:
			// F=ctable(A,1,W)
			wtBlock = (MatrixBlock) ec.getMatrixInput(input3.get_name());
			cst1 = ec.getScalarInput(input2.get_name(), input2.get_valueType(), input2.isLiteral()).getDoubleValue();
			matBlock1.tertiaryOperations((SimpleOperator)optr, cst1, wtBlock, ctableMap);
			break;
		
		default:
			throw new DMLRuntimeException("Encountered an invalid ctable operation ("+ctableOp+") while executing instruction: " + this.toString());
		}
		
		matBlock1 = matBlock2 = wtBlock = null;
		
		if(input1.get_dataType() == DataType.MATRIX)
			ec.releaseMatrixInput(input1.get_name());
		if(input2.get_dataType() == DataType.MATRIX)
			ec.releaseMatrixInput(input2.get_name());
		if(input3.get_dataType() == DataType.MATRIX)
			ec.releaseMatrixInput(input3.get_name());
		
		resultBlock = DataConverter.convertToMatrixBlock( ctableMap );
		ec.setMatrixOutput(output.get_name(), resultBlock);
		ctableMap.clear();
		resultBlock = null;
	}	
}
