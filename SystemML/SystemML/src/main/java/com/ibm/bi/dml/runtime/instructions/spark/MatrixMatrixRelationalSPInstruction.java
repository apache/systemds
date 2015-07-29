/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.functions.MatrixMatrixBinaryOpFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ReplicateVector;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class MatrixMatrixRelationalSPInstruction extends RelationalBinarySPInstruction {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public MatrixMatrixRelationalSPInstruction(Operator op, 
											   CPOperand in1, 
											   CPOperand in2, 
											   CPOperand out,
											   String opcode,
											   String istr){
		super(op, in1, in2, out, opcode, istr);
	}

	private String getReplicatedVar(String rddVar1, String rddVar2, boolean isRowVectorOperation, MatrixCharacteristics mc1) {
		String replicatedVar = rddVar2;
		if(isRowVectorOperation) {
			if(mc1.getCols() == 1) {
				replicatedVar = rddVar1;
			}
		}
		else {
			if(mc1.getRows() == 1) {
				replicatedVar = rddVar1;
			}
		}
		return replicatedVar;
	}
	
	private long getNumReplications(boolean isRowVectorOperation, MatrixCharacteristics otherMc) {
		long numReplications;
		if(isRowVectorOperation) {
			numReplications = (long) Math.ceil(otherMc.getRows() / otherMc.getRowsPerBlock());
		}
		else {
			numReplications = (long) Math.ceil(otherMc.getCols() / otherMc.getColsPerBlock());
		}
		return numReplications;
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException{
		if(input1.getDataType() == DataType.MATRIX && input2.getDataType() == DataType.MATRIX) {
			String opcode = getOpcode();
			if ( opcode.equalsIgnoreCase("==") || opcode.equalsIgnoreCase("!=") || opcode.equalsIgnoreCase("<")
					|| opcode.equalsIgnoreCase(">") || opcode.equalsIgnoreCase("<=") || opcode.equalsIgnoreCase(">=")) { 
				
				SparkExecutionContext sec = (SparkExecutionContext)ec;
				
				// Get input RDDs
				String rddVar1 = input1.getName(); 
				String rddVar2 = input2.getName();
				
				MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(rddVar1);
				MatrixCharacteristics mc2 = sec.getMatrixCharacteristics(rddVar2);
				
				if(mc1.getRowsPerBlock() != mc2.getRowsPerBlock() ||  mc1.getColsPerBlock() != mc2.getColsPerBlock()) {
					throw new DMLRuntimeException("MatrixMatrixRelationalSPinstruction is only supported for matrices with same blocksizes "
							+ "[(" + mc1.getRowsPerBlock() + "," + mc1.getColsPerBlock()  + "), (" + mc2.getRowsPerBlock() + "," + mc2.getColsPerBlock() + ")]");
				}
				
				JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = null;
				JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = null;
				JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
				BinaryOperator bop = (BinaryOperator) _optr;
				
				boolean isOuter = (mc1.getRows() != 1 && mc2.getCols() != 1 && mc1.getCols() == 1 && mc2.getRows() == 1);
				if(isOuter) {
					// Outer operation
					if(mc1.getRowsPerBlock() != mc2.getRowsPerBlock() || mc1.getColsPerBlock() != mc2.getColsPerBlock()) {
						throw new DMLRuntimeException("Outer for incompatible block sizes is not implemented");
					}
					in1 = sec.getBinaryBlockRDDHandleForVariable( rddVar1 )
							.flatMapToPair(new ReplicateVector(false, mc2.getCols() ));
					in2 = sec.getBinaryBlockRDDHandleForVariable( rddVar2 )
							.flatMapToPair(new ReplicateVector(true, mc1.getRows()));
				}
				else if(mc1.getRows() == mc2.getRows() && mc1.getCols() == mc2.getCols()) {
					// Matrix-matrix operation
					in1 = sec.getBinaryBlockRDDHandleForVariable( rddVar1 );
					in2 = sec.getBinaryBlockRDDHandleForVariable( rddVar2 );
				}
				else {
					// Matrix vector operation where vector doesnot fit in broadcast memory
					// Replicate the vector
					// Conservative approach: so as not to crash when matrix is large. 
					// This can be improved further when we know number of blocks (of non-replicated dimension) fit into task limit
					// by cogrouping the entire row/column of matrix with that of vector
					boolean atLeastOneColumnVector = mc1.getCols() == 1 || mc2.getCols() == 1;
					boolean isColumnVectorOperation = mc1.getRows() == mc2.getRows() && mc1.getCols() != mc2.getCols() && atLeastOneColumnVector;
					boolean atLeastOneRowVector = mc1.getRows() == 1 || mc2.getRows() == 1;
					boolean isRowVectorOperation = mc1.getRows() != mc2.getRows() && mc1.getCols() == mc2.getCols() && atLeastOneRowVector;
					if(!isColumnVectorOperation && !isRowVectorOperation) {
						throw new DMLRuntimeException("Incorrect input dimensions for MatrixMatrixRelationalSPinstruction");
					}
					
					String replicatedVar = getReplicatedVar(rddVar1,rddVar2, isRowVectorOperation, mc1);
					
					if(rddVar1.compareTo(replicatedVar) == 0) {
						long numReplications = getNumReplications(isRowVectorOperation, mc2);
						in1 = sec.getBinaryBlockRDDHandleForVariable( rddVar1 )
								.flatMapToPair(new ReplicateVector(isRowVectorOperation, numReplications));
					}
					else {
						in1 = sec.getBinaryBlockRDDHandleForVariable( rddVar1 );
					}
					
					if(rddVar2.compareTo(replicatedVar) == 0) {
						long numReplications = getNumReplications(isRowVectorOperation, mc1);
						in2 = sec.getBinaryBlockRDDHandleForVariable( rddVar2 )
								.flatMapToPair(new ReplicateVector(isRowVectorOperation, numReplications));
					}
					else {
						in2 = sec.getBinaryBlockRDDHandleForVariable( rddVar2 );
					}
				}
				
				// Core operation: perform cogroup and then map
				out = in1.join(in2).mapValues(new MatrixMatrixBinaryOpFunction(bop));
				
				//put output RDD handle into symbol table
				MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
				if(!mcOut.dimsKnown()) {
					if(!mc1.dimsKnown())
						throw new DMLRuntimeException("The output dimensions are not specified for MatrixMatrixRelationalSPInstruction");
					else if(isOuter) {
						sec.getMatrixCharacteristics(output.getName()).set(mc1.getRows(), mc2.getCols(), mc1.getRowsPerBlock(), mc1.getColsPerBlock());
					}
					else {
						sec.getMatrixCharacteristics(output.getName()).set(mc1);
					}
				}
				
				sec.setRDDHandleForVariable(output.getName(), out);
				sec.addLineageRDD(output.getName(), rddVar1);
				sec.addLineageRDD(output.getName(), rddVar2);
			}
			else {
				throw new DMLRuntimeException("Unknown opcode in MatrixMatrixRelationalSPInstruction: " + toString());
			}
		}
		else {
			throw new DMLRuntimeException("MatrixMatrixRelationalSPInstruction is only applicable for matrix inputs");
		}
	}
}
