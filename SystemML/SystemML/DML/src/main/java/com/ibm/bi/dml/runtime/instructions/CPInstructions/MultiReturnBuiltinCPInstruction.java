/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.util.DataConverter;


public class MultiReturnBuiltinCPInstruction extends ComputationCPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
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
	
	public CPOperand getOutput(int i)
	{
		return _outputs.get(i);
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
		else if ( opcode.equalsIgnoreCase("lu") ) {
			CPOperand in1 = new CPOperand(parts[1]);
			
			// one input and three outputs
			String[] operandParts = parts[2].split(Lop.SEPARATOR_WITHIN_OPRAND);
			outputs.add ( new CPOperand(operandParts[0], ValueType.DOUBLE, DataType.MATRIX) );
			outputs.add ( new CPOperand(operandParts[1], ValueType.DOUBLE, DataType.MATRIX) );
			outputs.add ( new CPOperand(operandParts[2], ValueType.DOUBLE, DataType.MATRIX) );
			
			return new MultiReturnBuiltinCPInstruction(null, in1, outputs, str);
			
		}
		else if ( opcode.equalsIgnoreCase("eigen") ) {
			CPOperand in1 = new CPOperand(parts[1]);
			
			// one input and two outputs
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
			performQR(ec);
		}
		else if ( opcode.equalsIgnoreCase("lu") ) {
			performLU(ec);
		}
		else if ( opcode.equalsIgnoreCase("eigen") ) {
			performEigen(ec);
		}
		else {
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + opcode);
		}
	}

	private void performLU(ExecutionContext ec) throws DMLRuntimeException {
		// Prepare input to commons math function
		MatrixObject mobjInput = (MatrixObject) ec.getVariable(input1.get_name());
		if ( mobjInput.getNumRows() != mobjInput.getNumColumns() ) {
			throw new DMLRuntimeException("LU Decomposition can only be done on a square matrix. Input matrix is rectangular (rows=" + mobjInput.getNumRows() + ", cols="+ mobjInput.getNumColumns() +")");
		}
		Array2DRowRealMatrix matrixInput = DataConverter.convertToArray2DRowRealMatrix((MatrixObject)ec.getVariable(input1.get_name()));
		
		// Perform LUP decomposition
		LUDecomposition ludecompose = new LUDecomposition(matrixInput);
		RealMatrix P = ludecompose.getP();
		RealMatrix L = ludecompose.getL();
		RealMatrix U = ludecompose.getU();
		
		// Read the results into native format
		MatrixBlock mbP = DataConverter.convertToMatrixBlock(P.getData());
		MatrixBlock mbL = DataConverter.convertToMatrixBlock(L.getData());
		MatrixBlock mbU = DataConverter.convertToMatrixBlock(U.getData());
		
		ec.setMatrixOutput(_outputs.get(0).get_name(), mbP);
		ec.setMatrixOutput(_outputs.get(1).get_name(), mbL);
		ec.setMatrixOutput(_outputs.get(2).get_name(), mbU);
		//ec.releaseMatrixInput(input1.get_name());
	}
	
	private void performQR(ExecutionContext ec) throws DMLRuntimeException {
		// Prepare input to commons math function
		Array2DRowRealMatrix matrixInput = DataConverter.convertToArray2DRowRealMatrix((MatrixObject)ec.getVariable(input1.get_name()));
		
		// Perform QR decomposition
		QRDecomposition qrdecompose = new QRDecomposition(matrixInput);
		RealMatrix H = qrdecompose.getH();
		RealMatrix R = qrdecompose.getR();
		
		// Read the results into native format
		MatrixBlock mbH = DataConverter.convertToMatrixBlock(H.getData());
		MatrixBlock mbR = DataConverter.convertToMatrixBlock(R.getData());
		
		ec.setMatrixOutput(_outputs.get(0).get_name(), mbH);
		ec.setMatrixOutput(_outputs.get(1).get_name(), mbR);
	}
	
	/**
	 * Helper function to perform eigen decomposition using commons-math library.
	 * Input matrix must be a symmetric matrix.
	 * 
	 * @param ec
	 * @throws DMLRuntimeException
	 */
	private void performEigen(ExecutionContext ec) throws DMLRuntimeException {
		MatrixObject mobjInput = (MatrixObject) ec.getVariable(input1.get_name());
		if ( mobjInput.getNumRows() != mobjInput.getNumColumns() ) {
			throw new DMLRuntimeException("Eigen Decomposition can only be done on a square matrix. Input matrix is rectangular (rows=" + mobjInput.getNumRows() + ", cols="+ mobjInput.getNumColumns() +")");
		}
		Array2DRowRealMatrix matrixInput = DataConverter.convertToArray2DRowRealMatrix((MatrixObject)ec.getVariable(input1.get_name()));
		
		EigenDecomposition eigendecompose = new EigenDecomposition(matrixInput, 0.0);
		RealMatrix eVectorsMatrix = eigendecompose.getV();
		double[][] eVectors = eVectorsMatrix.getData();
		double[] eValues = eigendecompose.getRealEigenvalues();
		
		//Sort the eigen values (and vectors) in increasing order (to be compatible w/ LAPACK.DSYEVR())
		int n = eValues.length;
		for (int i = 0; i < n; i++) {
		    int k = i;
		    double p = eValues[i];
		    for (int j = i + 1; j < n; j++) {
		        if (eValues[j] < p) {
		            k = j;
		            p = eValues[j];
		        }
		    }
		    if (k != i) {
		        eValues[k] = eValues[i];
		        eValues[i] = p;
		        for (int j = 0; j < n; j++) {
		            p = eVectors[j][i];
		            eVectors[j][i] = eVectors[j][k];
		            eVectors[j][k] = p;
		        }
		    }
		}

		MatrixBlock mbValues = DataConverter.convertToMatrixBlock(eValues, true);
		MatrixBlock mbVectors = DataConverter.convertToMatrixBlock(eVectors);
		
		ec.setMatrixOutput(_outputs.get(0).get_name(), mbValues);
		ec.setMatrixOutput(_outputs.get(1).get_name(), mbVectors);
	}

}
