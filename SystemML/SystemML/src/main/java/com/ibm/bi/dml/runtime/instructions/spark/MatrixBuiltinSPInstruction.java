package com.ibm.bi.dml.runtime.instructions.spark;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.UnaryOperator;

public class MatrixBuiltinSPInstruction  extends BuiltinUnarySPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public MatrixBuiltinSPInstruction(Operator op,
									  CPOperand in,
									  CPOperand out,
									  String opcode,
									  String instr){
		super(op, in, out, 1, opcode, instr);
	}

	@Override 
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{	
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		try {
		    PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("myfile.txt", true)));
		    out.println("MatrixBuiltinSPInstruction:"+ getOpcode());
		    out.close();
		} catch (IOException e) {
		    //exception handling left as an exercise for the reader
		}
		
		//get input
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockedRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1.mapToPair(new RDDMatrixBuiltinUnaryOp(getOpcode(), (UnaryOperator) _optr));
		
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		if(!mcOut.dimsKnown()) {
			throw new DMLRuntimeException("The output dimensions are not specified for MatrixBuiltinSPInstruction");
		}
		
		//set output RDD
		sec.setRDDHandleForVariable(output.getName(), out);	
		sec.addLineageRDD(output.getName(), input1.getName());
	}
	
	public static class RDDMatrixBuiltinUnaryOp implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock> {
		private static final long serialVersionUID = -3128192099832877491L;
		String opcode;
		UnaryOperator u_op;
		public RDDMatrixBuiltinUnaryOp(String opcode, UnaryOperator u_op) {
			this.opcode = opcode;
			this.u_op = u_op;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
			MatrixBlock resultBlock = null;
			// Cannot execute LibCommonsMath in Spark
//			if(LibCommonsMath.isSupportedUnaryOperation(opcode)) {
//				resultBlock = LibCommonsMath.unaryOperations((MatrixObject) kv._2, opcode);
//			}
//			else {
				resultBlock = (MatrixBlock) (kv._2.unaryOperations(u_op, new MatrixBlock()));
//			}
			return new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, resultBlock);
		}
		
	}
}

