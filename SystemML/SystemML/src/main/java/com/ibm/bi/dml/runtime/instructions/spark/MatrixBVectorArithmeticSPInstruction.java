package com.ibm.bi.dml.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.broadcast.Broadcast;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.data.PartitionedMatrixBlock;
import com.ibm.bi.dml.runtime.instructions.spark.functions.MatrixVectorBinaryOpFunction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class MatrixBVectorArithmeticSPInstruction extends ArithmeticBinarySPInstruction {

	public MatrixBVectorArithmeticSPInstruction(Operator op,
			CPOperand in1, CPOperand in2, CPOperand out, String opcode,
			String istr) {
		super(op, in1, in2, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException, DMLUnsupportedOperationException {
		if(input1.getDataType() == DataType.MATRIX && input2.getDataType() == DataType.MATRIX) {
			String opcode = getOpcode();
			if ( opcode.equalsIgnoreCase("map+") || opcode.equalsIgnoreCase("map-") || opcode.equalsIgnoreCase("map*")
				|| opcode.equalsIgnoreCase("map/") || opcode.equalsIgnoreCase("map%%") || opcode.equalsIgnoreCase("map%/%")
				|| opcode.equalsIgnoreCase("map^") ) { 
				// || opcode.equalsIgnoreCase("^2") || opcode.equalsIgnoreCase("*2")) {
				SparkExecutionContext sec = (SparkExecutionContext)ec;
				
				// Get input RDDs
				String rddVar1 = input1.getName(); 
				String rddVar2 = input2.getName();
				
				MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(rddVar1);
				MatrixCharacteristics mc2 = sec.getMatrixCharacteristics(rddVar2);
				
				if(mc1.getRowsPerBlock() != mc2.getRowsPerBlock() ||  mc1.getColsPerBlock() != mc2.getColsPerBlock()) {
					throw new DMLRuntimeException("MatrixBVectorArithmeticSPInstruction is only supported for matrices with same blocksizes "
							+ "[(" + mc1.getRowsPerBlock() + "," + mc1.getColsPerBlock()  + "), (" + mc2.getRowsPerBlock() + "," + mc2.getColsPerBlock() + ")]");
				}
				 
				
				JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
				BinaryOperator bop = (BinaryOperator) _optr;
				
				boolean isBroadcastRHSVar = true;
				
				// Matrix-column vector operation
				boolean atLeastOneColumnVector = mc1.getCols() == 1 || mc2.getCols() == 1;
				boolean isColumnVectorOperation = mc1.getRows() == mc2.getRows() && atLeastOneColumnVector;
				boolean atLeastOneRowVector = mc1.getRows() == 1 || mc2.getRows() == 1;
				boolean isRowVectorOperation = mc1.getCols() == mc2.getCols() && atLeastOneRowVector;
				boolean isOuter = (mc1.getRows() != 1 && mc2.getCols() != 1 && mc1.getCols() == 1 && mc2.getRows() == 1);
				if(!isColumnVectorOperation && !isRowVectorOperation && !isOuter) {
					throw new DMLRuntimeException("Incorrect input dimensions for MatrixBVectorArithmeticSPInstruction");
				}
				
				String rddVar = rddVar1; 
				String bcastVar = rddVar2;
				MatrixCharacteristics rddMC = mc1;
				if(isRowVectorOperation) {
					if(mc1.getCols() == 1) {
						isBroadcastRHSVar = false; rddVar = rddVar2;  bcastVar = rddVar1;
						rddMC = mc2;
					}
				}
				else {
					if(mc1.getRows() == 1) {
						isBroadcastRHSVar = false; rddVar = rddVar2;  bcastVar = rddVar1;
						rddMC = mc2;
					}
				}
				
				JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( rddVar );
				Broadcast<PartitionedMatrixBlock> in2 = sec.getBroadcastForVariable( bcastVar );
				out = in1.flatMapToPair(new MatrixVectorBinaryOpFunction(isBroadcastRHSVar, isColumnVectorOperation, in2, bop, rddMC.getRowsPerBlock(), rddMC.getColsPerBlock(), rddMC.getRows(), rddMC.getCols(), isOuter));
				
				// put output RDD handle into symbol table
				MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
				if(!mcOut.dimsKnown()) {
					if(!mc1.dimsKnown())
						throw new DMLRuntimeException("The output dimensions are not specified for MatrixBVectorArithmeticSPInstruction");
					else if(isOuter) {
						sec.getMatrixCharacteristics(output.getName()).set(mc1.getRows(), mc2.getCols(), mc1.getRowsPerBlock(), mc1.getColsPerBlock());
					}
					else {
						sec.getMatrixCharacteristics(output.getName()).set(mc1);
					}
				}
				
				sec.setRDDHandleForVariable(output.getName(), out);
				sec.addLineageRDD(output.getName(), rddVar1);
				if( isBroadcastRHSVar )
					sec.addLineageBroadcast(output.getName(), rddVar2);
				else
					sec.addLineageRDD(output.getName(), rddVar2);
			}
			else {
				throw new DMLRuntimeException("Unknown opcode in MatrixBVectorArithmeticSPInstruction: " + toString());
			}
		}
		else {
			throw new DMLRuntimeException("MatrixBVectorArithmeticSPInstruction is only applicable for matrix inputs");
		}
		
	}
}
