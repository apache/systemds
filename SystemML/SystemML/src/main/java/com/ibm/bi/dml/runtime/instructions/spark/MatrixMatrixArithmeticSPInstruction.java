/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import java.util.Iterator;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.data.PartitionedMatrixBlock;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class MatrixMatrixArithmeticSPInstruction extends ArithmeticBinarySPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public MatrixMatrixArithmeticSPInstruction(Operator op, 
											   CPOperand in1, 
											   CPOperand in2, 
											   CPOperand out, 
											   String opcode,
											   String istr){
		super(op, in1, in2, out, opcode, istr);
	}
	
//	public String outputMatrixIndexes(List<Tuple2<MatrixIndexes, MatrixBlock>> vals) {
//		String retVal = "{";
//		for(Tuple2<MatrixIndexes, MatrixBlock> kv : vals) {
//			retVal += kv._1.toString() + " ";
//		}
//		retVal += "}";
//		return retVal;
//	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		if(input1.getDataType() == DataType.MATRIX && input2.getDataType() == DataType.MATRIX) {
			String opcode = getOpcode();
			if ( opcode.equalsIgnoreCase("+") || opcode.equalsIgnoreCase("-") || opcode.equalsIgnoreCase("*")
				|| opcode.equalsIgnoreCase("/") || opcode.equalsIgnoreCase("%%") || opcode.equalsIgnoreCase("%/%")
				|| opcode.equalsIgnoreCase("^") ) { 
				// || opcode.equalsIgnoreCase("^2") || opcode.equalsIgnoreCase("*2")) {
				SparkExecutionContext sec = (SparkExecutionContext)ec;
				
				// Get input RDDs
				String rddVar1 = input1.getName(); 
				String rddVar2 = input2.getName();
				
				MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(rddVar1);
				MatrixCharacteristics mc2 = sec.getMatrixCharacteristics(rddVar2);
				
				if(mc1.getRowsPerBlock() != mc2.getRowsPerBlock() ||  mc1.getColsPerBlock() != mc2.getColsPerBlock()) {
					throw new DMLRuntimeException("MatrixMatrixArithmeticSPinstruction is only supported for matrices with same blocksizes "
							+ "[(" + mc1.getRowsPerBlock() + "," + mc1.getColsPerBlock()  + "), (" + mc2.getRowsPerBlock() + "," + mc2.getColsPerBlock() + ")]");
				}
				 
				
				JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
				BinaryOperator bop = (BinaryOperator) _optr;
				boolean isBroadcastRHSVar = true;
				
				if(mc1.getRows() == mc2.getRows() && mc1.getCols() == mc2.getCols()) {
					// Matrix-matrix operation
					JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( rddVar1 );
					JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryBlockRDDHandleForVariable( rddVar2 );
					JavaPairRDD<MatrixIndexes, Tuple2<Iterable<MatrixBlock>, Iterable<MatrixBlock>>> cogroupRdd = in1.cogroup(in2);
					out = cogroupRdd.mapToPair(new RDDMatrixMatrixArithmeticFunction(bop, this.instString));
					isBroadcastRHSVar = false;
				}
				else {
					// Matrix-column vector operation
					boolean atLeastOneColumnVector = mc1.getCols() == 1 || mc2.getCols() == 1;
					boolean isColumnVectorOperation = mc1.getRows() == mc2.getRows() && mc1.getCols() != mc2.getCols() && atLeastOneColumnVector;
					boolean atLeastOneRowVector = mc1.getRows() == 1 || mc2.getRows() == 1;
					boolean isRowVectorOperation = mc1.getRows() != mc2.getRows() && mc1.getCols() == mc2.getCols() && atLeastOneRowVector;
					if(!isColumnVectorOperation && !isRowVectorOperation) {
						throw new DMLRuntimeException("Incorrect input dimensions for MatrixMatrixArithmeticSPInstruction");
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
					out = in1.mapToPair(new RDDMatrixVectorArithmeticFunction(isBroadcastRHSVar, isColumnVectorOperation, in2, bop, rddMC.getRowsPerBlock(), rddMC.getColsPerBlock(), rddMC.getRows(), rddMC.getCols()));
				}
				
				// put output RDD handle into symbol table
				MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
				if(!mcOut.dimsKnown()) {
					if(!mc1.dimsKnown())
						throw new DMLRuntimeException("The output dimensions are not specified for MatrixMatrixArithmeticSPInstruction");
					else
						sec.getMatrixCharacteristics(output.getName()).set(mc1);
				}
				
				sec.setRDDHandleForVariable(output.getName(), out);
				sec.addLineageRDD(output.getName(), rddVar1);
				if( isBroadcastRHSVar )
					sec.addLineageBroadcast(output.getName(), rddVar2);
				else
					sec.addLineageRDD(output.getName(), rddVar2);
			}
			else {
				throw new DMLRuntimeException("Unknown opcode in MatrixMatrixArithmeticSPInstruction: " + toString());
			}
		}
		else {
			throw new DMLRuntimeException("MatrixMatrixArithmeticSPInstruction is only applicable for matrix inputs");
		}
	}
	
	private static class RDDMatrixMatrixArithmeticFunction implements PairFunction<Tuple2<MatrixIndexes, Tuple2<Iterable<MatrixBlock>, Iterable<MatrixBlock>>>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 8197406787010296291L;
		private BinaryOperator op;
		
		private String instructionString;
		public RDDMatrixMatrixArithmeticFunction(BinaryOperator op, String instructionString) {
			this.op = op;
			this.instructionString = instructionString;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, Tuple2<Iterable<MatrixBlock>, Iterable<MatrixBlock>>> kv) throws Exception {
			// MatrixBlock resultBlk = new MatrixBlock(brlen, bclen, false);
			Iterator<MatrixBlock> iter1 = kv._2._1.iterator();
			MatrixBlock blk1 = null;
			if(iter1.hasNext()) {
				blk1 = iter1.next();
			}
			Iterator<MatrixBlock> iter2 = kv._2._2.iterator();
			MatrixBlock blk2 = null;
			if(iter2.hasNext()) {
				blk2 = iter2.next();
			}
			
			if(blk1 == null && blk2 == null) {
				throw new Exception("Error: In instruction:[" + instructionString + "]: The input variables donot have blocks for index:"+ kv._1.toString());
			}
			else if(blk1 == null) {
				throw new Exception("Error: In instruction:[" + instructionString + "]: The LHS variable doesnot have a block for index:"+ kv._1.toString());
			}
			else if(blk2 == null) {
				throw new Exception("Error: In instruction:[" + instructionString + "]: The RHS variable doesnot have a block for index:"+ kv._1.toString());
			}
			else if(iter1.hasNext() || iter2.hasNext()) {
				throw new Exception("Error: In instruction:[" + instructionString + "]: The iterator for RDDMatrixMatrixArithmeticFunction should be size of 1.");
			}
			
			MatrixBlock resultBlk = (MatrixBlock) (blk1.binaryOperations (op, blk2, new MatrixBlock()));
			// LibMatrixBincell.bincellOp(blk1, blk2, resultBlk, op);
			
			return new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, resultBlk);
		}
	}
	
	private static class RDDMatrixVectorArithmeticFunction implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock> {
		private static final long serialVersionUID = 2814994742189295067L;
		private BinaryOperator op;
		private boolean isBroadcastRHSVar;
		private boolean isColumnVector; 
		Broadcast<PartitionedMatrixBlock> _pmV;
		
		public RDDMatrixVectorArithmeticFunction( boolean isBroadcastRHSVar, boolean isColumnVector, Broadcast<PartitionedMatrixBlock> binput, BinaryOperator op, int brlen, int bclen, long rdd_rlen, long rdd_clen ) {
			this.isBroadcastRHSVar = isBroadcastRHSVar;
			this.isColumnVector = isColumnVector;
			this.op = op;
			
			//get the broadcast vector
			_pmV = binput;
		}
		
		
		@Override
		public Tuple2<MatrixIndexes,MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> kv) 
			throws Exception 
		{
			MatrixBlock blk1 = null;
			MatrixBlock blk2 = null;
			
			if(isBroadcastRHSVar) {
				blk1 = kv._2;
				if(this.isColumnVector) {
					blk2 = _pmV.value().getMatrixBlock((int) kv._1.getRowIndex(),1);
				}
				else {
					blk2 = _pmV.value().getMatrixBlock(1, (int) kv._1.getColumnIndex());
				}
			}
			else {
				blk2 = kv._2;
				if(this.isColumnVector) {
					blk2 = _pmV.value().getMatrixBlock((int) kv._1.getRowIndex(),1);
				}
				else {
					blk2 = _pmV.value().getMatrixBlock(1, (int) kv._1.getColumnIndex());
				}
			}
			
			MatrixBlock resultBlk = (MatrixBlock) (blk1.binaryOperations (op, blk2, new MatrixBlock()));
			// LibMatrixBincell.bincellOp(blk1, blk2, resultBlk, op);
			
			return new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, resultBlk);
		}
		
	}
}