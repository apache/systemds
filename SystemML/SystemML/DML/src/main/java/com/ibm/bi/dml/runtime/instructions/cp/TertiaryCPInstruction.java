/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import java.util.HashMap;

import com.ibm.bi.dml.lops.Tertiary;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;
import com.ibm.bi.dml.runtime.util.DataConverter;


public class TertiaryCPInstruction extends ComputationCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private String _outDim1;
	private String _outDim2;
	private boolean _dim1Literal; 
	private boolean _dim2Literal;
	private boolean _isExpand;
	
	public TertiaryCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, 
							 String outputDim1, boolean dim1Literal,String outputDim2, boolean dim2Literal, 
							 boolean isExpand, String istr )
	{
		super(op, in1, in2, in3, out);
		_outDim1 = outputDim1;
		_dim1Literal = dim1Literal;
		_outDim2 = outputDim2;
		_dim2Literal = dim2Literal;
		_isExpand = isExpand;
		
		instString = istr;
	}

	public static TertiaryCPInstruction parseInstruction(String inst) throws DMLRuntimeException{
		
		InstructionUtils.checkNumFields ( inst, 6 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(inst);
		String opcode = parts[0];
		
		//handle opcode
		if ( !(opcode.equalsIgnoreCase("ctable") || opcode.equalsIgnoreCase("ctableexpand")) ) {
			throw new DMLRuntimeException("Unexpected opcode in TertiaryCPInstruction: " + inst);
		}
		boolean isExpand = opcode.equalsIgnoreCase("ctableexpand");
		
		//handle operands
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		
		//handle known dimension information
		String[] dim1Fields = parts[4].split(Instruction.LITERAL_PREFIX);
		String[] dim2Fields = parts[5].split(Instruction.LITERAL_PREFIX);

		CPOperand out = new CPOperand(parts[6]);
		
		// ctable does not require any operator, so we simply pass-in a dummy operator with null functionobject
		return new TertiaryCPInstruction(new SimpleOperator(null), in1, in2, in3, out, dim1Fields[0], Boolean.parseBoolean(dim1Fields[1]), dim2Fields[0], Boolean.parseBoolean(dim2Fields[1]), isExpand, inst);
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
		
		MatrixBlock matBlock1 = ec.getMatrixInput(input1.get_name());
		MatrixBlock matBlock2=null, wtBlock=null;
		double cst1, cst2;
		
		HashMap<MatrixIndexes,Double> ctableMap = new HashMap<MatrixIndexes,Double>();
		MatrixBlock resultBlock = null;
		Tertiary.OperationTypes ctableOp = findCtableOperation();
		ctableOp = _isExpand ? Tertiary.OperationTypes.CTABLE_EXPAND_SCALAR_WEIGHT : ctableOp;
		
		long _outputDim1 = (_dim1Literal ? (long) Double.parseDouble(_outDim1) : (ec.getScalarInput(_outDim1, ValueType.DOUBLE, false)).getLongValue());
		long _outputDim2 = (_dim2Literal ? (long) Double.parseDouble(_outDim2) : (ec.getScalarInput(_outDim2, ValueType.DOUBLE, false)).getLongValue());
		
		boolean outputDimsKnown = (_outputDim1 != -1 && _outputDim2 != -1);
		if ( outputDimsKnown ) {
			int inputRows = matBlock1.getNumRows();
			boolean sparse = (inputRows < _outputDim1*_outputDim2);
			resultBlock = new MatrixBlock((int)_outputDim1, (int)_outputDim2, sparse);
		}
		if( _isExpand ){
			resultBlock = new MatrixBlock( matBlock1.getNumRows(), Integer.MAX_VALUE, true );
		}
		
		switch(ctableOp) {
		case CTABLE_TRANSFORM:
			// F=ctable(A,B,W)
			matBlock2 = ec.getMatrixInput(input2.get_name());
			wtBlock = ec.getMatrixInput(input3.get_name());
			matBlock1.tertiaryOperations((SimpleOperator)optr, matBlock2, wtBlock, ctableMap, resultBlock);
			break;
		case CTABLE_TRANSFORM_SCALAR_WEIGHT:
			// F = ctable(A,B) or F = ctable(A,B,1)
			matBlock2 = ec.getMatrixInput(input2.get_name());
			cst1 = ec.getScalarInput(input3.get_name(), input3.get_valueType(), input3.isLiteral()).getDoubleValue();
			matBlock1.tertiaryOperations((SimpleOperator)optr, matBlock2, cst1, ctableMap, resultBlock);
			break;
		case CTABLE_EXPAND_SCALAR_WEIGHT:
			// F = ctable(seq,A) or F = ctable(seq,B,1)
			matBlock2 = ec.getMatrixInput(input2.get_name());
			cst1 = ec.getScalarInput(input3.get_name(), input3.get_valueType(), input3.isLiteral()).getDoubleValue();
			// only resultBlock.rlen known, resultBlock.clen set in operation
			matBlock1.tertiaryOperations((SimpleOperator)optr, matBlock2, cst1, resultBlock);
			break;
		case CTABLE_TRANSFORM_HISTOGRAM:
			// F=ctable(A,1) or F = ctable(A,1,1)
			cst1 = ec.getScalarInput(input2.get_name(), input2.get_valueType(), input2.isLiteral()).getDoubleValue();
			cst2 = ec.getScalarInput(input3.get_name(), input3.get_valueType(), input3.isLiteral()).getDoubleValue();
			matBlock1.tertiaryOperations((SimpleOperator)optr, cst1, cst2, ctableMap, resultBlock);
			break;
		case CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM:
			// F=ctable(A,1,W)
			wtBlock = ec.getMatrixInput(input3.get_name());
			cst1 = ec.getScalarInput(input2.get_name(), input2.get_valueType(), input2.isLiteral()).getDoubleValue();
			matBlock1.tertiaryOperations((SimpleOperator)optr, cst1, wtBlock, ctableMap, resultBlock);
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
		
		if ( resultBlock == null )
			resultBlock = DataConverter.convertToMatrixBlock( ctableMap );
		else
			resultBlock.examSparsity();
		
		ec.setMatrixOutput(output.get_name(), resultBlock);
		ctableMap.clear();
		resultBlock = null;
	}	
}
