/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.QRDecompositionImpl;
import org.apache.commons.math.linear.RealMatrix;

import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.packagesupport.Matrix;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.util.DataConverter;


public class MultiReturnBuiltinCPInstruction extends ComputationCPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	int arity;
	protected ArrayList<CPOperand> _outputs;
	
	public MultiReturnBuiltinCPInstruction(Operator op, CPOperand input1, ArrayList<CPOperand> outputs, String istr )
	{
		super(op, input1, null, outputs.get(0));
		cptype = CPINSTRUCTION_TYPE.MultiReturnBuiltin;
		_outputs = outputs;
		instString = istr;
	}

	public int getArity() {
		return arity;
	}
	
	public static HashMap<String, String> constructParameterMap(String[] params) {
		// process all elements in "params" except first(opcode) and last(output)
		HashMap<String,String> paramMap = new HashMap<String,String>();
		
		// all parameters are of form <name=value>
		String[] parts;
		for ( int i=1; i <= params.length-2; i++ ) {
			parts = params[i].split(Lop.SEPARATOR_WITHIN_OPRAND);
			paramMap.put(parts[0], parts[1]);
		}
		
		return paramMap;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		ArrayList<CPOperand> outputs = new ArrayList<CPOperand>();
		// first part is always the opcode
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase("qr") ) {
			// one input and two ouputs
			CPOperand in1 = new CPOperand(parts[1]);
			
			String[] operandParts = parts[2].split(Lop.SEPARATOR_WITHIN_OPRAND);
			outputs.add ( new CPOperand(operandParts[0], ValueType.DOUBLE, DataType.MATRIX) );
			outputs.add ( new CPOperand(operandParts[1], ValueType.DOUBLE, DataType.MATRIX) );
			
			return new MultiReturnBuiltinCPInstruction(null, in1, outputs, str);
		}
		else {
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + opcode);
		}

	}

	@Override 
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException, DMLUnsupportedOperationException {
		String opcode = InstructionUtils.getOpCode(instString);
		
		if (opcode.equalsIgnoreCase("qr")) {
			// Prepare input to commons math function
			//long start = System.nanoTime();
			//long begin = start;
			MatrixObject mobjInput = (MatrixObject) ec.getVariable(input1.get_name());
			Matrix mathInput = new Matrix(mobjInput.getFileName(), mobjInput.getNumRows(), mobjInput.getNumColumns(), (mobjInput.getValueType() == ValueType.DOUBLE ? Matrix.ValueType.Double : Matrix.ValueType.Integer));
			mathInput.setMatrixObject(mobjInput);
			Array2DRowRealMatrix matrixInput = new Array2DRowRealMatrix(mathInput.getMatrixAsDoubleArray(), false);
			//long prep = System.nanoTime() - begin;
			
			// Perform QR decomposition
			//begin = System.nanoTime();
			QRDecompositionImpl qrdecompose = new QRDecompositionImpl(matrixInput);
			//long decompose = System.nanoTime() - begin;
			//begin = System.nanoTime();
			RealMatrix Q = qrdecompose.getQ();
			RealMatrix R = qrdecompose.getR();
			//long extract = System.nanoTime() - begin;
			
			//begin = System.nanoTime();
			// Read the results into native format
			MatrixBlock mbQ = DataConverter.convertToMatrixBlock(Q.getData());
			MatrixBlock mbR = DataConverter.convertToMatrixBlock(R.getData());
			//long prepResults = System.nanoTime() - begin;
			
			ec.setMatrixOutput(_outputs.get(0).get_name(), mbQ);
			ec.setMatrixOutput(_outputs.get(1).get_name(), mbR);
			//long stop = System.nanoTime() - start;
			//System.out.println("Total " + (stop*1e-6) + ": [" + (prep*1e-6) + ", " + (decompose*1e-6) + ", " + (extract*1e-6) + ", " + (prepResults*1e-6));
			
		}
		else {
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + opcode);
		}
	}
	

}
