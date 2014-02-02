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
import org.apache.commons.math.linear.EigenDecompositionImpl;
import org.apache.commons.math.linear.LUDecompositionImpl;
import org.apache.commons.math.linear.QRDecompositionImpl;
import org.apache.commons.math.linear.RealMatrix;

import com.ibm.bi.dml.lops.Lop;
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
		long start = System.nanoTime();
		long begin = start;
		
		MatrixObject mobjInput = (MatrixObject) ec.getVariable(input1.get_name());
		if ( mobjInput.getNumRows() != mobjInput.getNumColumns() ) {
			throw new DMLRuntimeException("LU Decomposition can only be done on a square matrix. Input matrix is rectangular (rows=" + mobjInput.getNumRows() + ", cols="+ mobjInput.getNumColumns() +")");
		}
		Array2DRowRealMatrix matrixInput = prepareInputForCommonsMath(ec, input1.get_name());
		long prep = System.nanoTime() - begin;
		
		// Perform LUP decomposition
		begin = System.nanoTime();
		LUDecompositionImpl ludecompose = new LUDecompositionImpl(matrixInput);
		long decompose = System.nanoTime() - begin;
		begin = System.nanoTime();
		RealMatrix P = ludecompose.getP();
		RealMatrix L = ludecompose.getL();
		RealMatrix U = ludecompose.getU();
		long extract = System.nanoTime() - begin;
		
		// Read the results into native format
		begin = System.nanoTime();
		MatrixBlock mbP = DataConverter.convertToMatrixBlock(P.getData());
		MatrixBlock mbL = DataConverter.convertToMatrixBlock(L.getData());
		MatrixBlock mbU = DataConverter.convertToMatrixBlock(U.getData());
		long prepResults = System.nanoTime() - begin;
		
		ec.setMatrixOutput(_outputs.get(0).get_name(), mbP);
		ec.setMatrixOutput(_outputs.get(1).get_name(), mbL);
		ec.setMatrixOutput(_outputs.get(2).get_name(), mbU);
		//ec.releaseMatrixInput(input1.get_name());
		long stop = System.nanoTime() - start;
		System.out.println("LUPTime " + (stop*1e-9) + ", " + (prep*1e-9) + ", " + (decompose*1e-9) + ", " + (extract*1e-9) + ", " + (prepResults*1e-9));
	}
	
	private void performQR(ExecutionContext ec) throws DMLRuntimeException {
		// Prepare input to commons math function
		//long start = System.nanoTime();
		//long begin = start;
		System.out.println("Processing QR...");
		Array2DRowRealMatrix matrixInput = prepareInputForCommonsMath(ec, input1.get_name());
		System.out.println("  Setup input");
		//long prep = System.nanoTime() - begin;
		
		// Perform QR decomposition
		//begin = System.nanoTime();
		QRDecompositionImpl qrdecompose = new QRDecompositionImpl(matrixInput);
		System.out.println("  Called constructor");
		//long decompose = System.nanoTime() - begin;
		//begin = System.nanoTime();
		RealMatrix H = qrdecompose.getH();
		RealMatrix R = qrdecompose.getR();
		System.out.println("  Got H and R");
		//long extract = System.nanoTime() - begin;
		
		// Read the results into native format
		//begin = System.nanoTime();
		MatrixBlock mbH = DataConverter.convertToMatrixBlock(H.getData());
		MatrixBlock mbR = DataConverter.convertToMatrixBlock(R.getData());
		System.out.println("  Converted H and R into MatrixBlock");
		//long prepResults = System.nanoTime() - begin;
		
		ec.setMatrixOutput(_outputs.get(0).get_name(), mbH);
		ec.setMatrixOutput(_outputs.get(1).get_name(), mbR);
		System.out.println("QR Done!");
		//ec.releaseMatrixInput(input1.get_name());
		//long stop = System.nanoTime() - start;
		//System.out.println("QRTime" + (stop*1e-9) + ", " + (prep*1e-9) + ", " + (decompose*1e-9) + ", " + (extract*1e-9) + ", " + (prepResults*1e-9));
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
		Array2DRowRealMatrix matrixInput = prepareInputForCommonsMath(ec, input1.get_name());
		
		EigenDecompositionImpl eigendecompose = new EigenDecompositionImpl(matrixInput, 0.0);
		RealMatrix eVectors = eigendecompose.getV();
		double[] eValues = eigendecompose.getRealEigenvalues();
		
		MatrixBlock mbVectors = DataConverter.convertToMatrixBlock(eVectors.getData());
		MatrixBlock mbValues = DataConverter.convertToMatrixBlock(eValues, true);
		
		/*RealMatrix Vt = eigendecompose.getVT();
		RealMatrix D = eigendecompose.getD();
		
		double[][] diff = eVectors.multiply(D).multiply(Vt).subtract(matrixInput).getData();
		double sum=0, count=0;
		for(int i=0; i < diff.length; i++) {
			for(int j=0; j< diff[i].length; j++) {
				sum += diff[i][j];
				if(diff[i][j] > 0) {
					System.out.println("    " + diff[i][j]);
					count++;
				}
			}
		}
		System.out.println("  --> " + sum + "  " + count);*/
		
		ec.setMatrixOutput(_outputs.get(0).get_name(), mbVectors);
		ec.setMatrixOutput(_outputs.get(1).get_name(), mbValues);
	}

}
