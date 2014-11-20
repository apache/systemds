/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.DecompositionSolver;
import org.apache.commons.math.linear.QRDecompositionImpl;
import org.apache.commons.math.linear.RealMatrix;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.util.DataConverter;


public class MatrixMatrixBuiltinCPInstruction extends BuiltinBinaryCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public MatrixMatrixBuiltinCPInstruction(Operator op, 
											   CPOperand in1, 
											   CPOperand in2, 
											   CPOperand out, 
											   String istr){
		super(op, in1, in2, out, 2, istr);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException{
        MatrixBlock matBlock1 = ec.getMatrixInput(input1.get_name());
        MatrixBlock matBlock2 = ec.getMatrixInput(input2.get_name());
        
        String opcode = InstructionUtils.getOpCode(instString);
        
        if ( opcode.equalsIgnoreCase("solve") ) {
        	executeSolve(ec);
        	return;
        }
		
        /* Default behavior of this instruction */
		String output_name = output.get_name();
		BinaryOperator bop = (BinaryOperator) optr;
		
		MatrixBlock resultBlock = (MatrixBlock) matBlock1.binaryOperations(bop, matBlock2, new MatrixBlock());
		
		ec.setMatrixOutput(output_name, resultBlock);
		
		resultBlock = matBlock1 = matBlock2 = null;
		ec.releaseMatrixInput(input1.get_name());
		ec.releaseMatrixInput(input2.get_name());
	}
	
	void executeSolve(ExecutionContext ec) 
		throws DMLRuntimeException 
	{
		Array2DRowRealMatrix matrixInput = prepareInputForCommonsMath(ec, input1.get_name());
		Array2DRowRealMatrix vectorInput = prepareInputForCommonsMath(ec, input2.get_name());
		
		/*LUDecompositionImpl ludecompose = new LUDecompositionImpl(matrixInput);
		DecompositionSolver lusolver = ludecompose.getSolver();
		RealMatrix solutionMatrix = lusolver.solve(vectorInput);*/
		
		// Setup a solver based on QR Decomposition
		QRDecompositionImpl qrdecompose = new QRDecompositionImpl(matrixInput);
		DecompositionSolver solver = qrdecompose.getSolver();
		// Invoke solve
		RealMatrix solutionMatrix = solver.solve(vectorInput);
		
		MatrixBlock solution = DataConverter.convertToMatrixBlock(solutionMatrix.getData());
		
		ec.setMatrixOutput(output.get_name(), solution);
		ec.releaseMatrixInput(input1.get_name());
		ec.releaseMatrixInput(input2.get_name());
		
		return;
	}
}