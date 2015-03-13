package com.ibm.bi.dml.runtime.instructions.spark;

import java.util.Iterator;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.LibMatrixBincell;
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
					throw new DMLRuntimeException("MatrixMatrixArithmeticSPinstruction is only supported for matrices with same blocksizes "
							+ "[(" + mc1.getRowsPerBlock() + "," + mc1.getColsPerBlock()  + "), (" + mc2.getRowsPerBlock() + "," + mc2.getColsPerBlock() + ")]");
				}
				
				JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getRDDHandleForVariable( rddVar1 );
				JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getRDDHandleForVariable( rddVar2 );
				
				JavaPairRDD<MatrixIndexes, Tuple2<Iterable<MatrixBlock>, Iterable<MatrixBlock>>> cogroupRdd = in1.cogroup(in2);
				
				BinaryOperator bop = (BinaryOperator) _optr;
				JavaPairRDD<MatrixIndexes,MatrixBlock> out = cogroupRdd.mapToPair(new RDDMatrixMatrixRelationalFunction(bop, mc1.getRowsPerBlock(), mc1.getColsPerBlock()));
				
				//put output RDD handle into symbol table
				sec.setRDDHandleForVariable(output.getName(), out);
			}
			else {
				throw new DMLRuntimeException("Unknown opcode in MatrixMatrixRelationalSPInstruction: " + toString());
			}
		}
		else {
			throw new DMLRuntimeException("MatrixMatrixRelationalSPInstruction is only applicable for matrix inputs");
		}
	}
	
	private static class RDDMatrixMatrixRelationalFunction implements PairFunction<Tuple2<MatrixIndexes, Tuple2<Iterable<MatrixBlock>, Iterable<MatrixBlock>>>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 8197406787010296291L;
		private int brlen; 
		private int bclen;
		private BinaryOperator op;
		
		public RDDMatrixMatrixRelationalFunction(BinaryOperator op, int brlen, int bclen) {
			this.op = op;
			this.brlen = brlen;
			this.bclen = bclen;
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
			
			if(blk1 == null || blk2 == null || iter1.hasNext() || iter2.hasNext()) {
				throw new Exception("The iterator for RDDMatrixMatrixRelationalFunction should be size of 1");
			}
			
			MatrixBlock resultBlk = (MatrixBlock) (blk1.binaryOperations (op, blk2, new MatrixBlock()));
			// LibMatrixBincell.bincellOp(blk1, blk2, resultBlk, op);
			
			return new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, resultBlk);
		}
	}
}
