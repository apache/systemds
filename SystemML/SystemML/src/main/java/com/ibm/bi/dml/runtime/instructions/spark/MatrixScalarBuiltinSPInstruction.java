/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.cp.ScalarObject;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;

public class MatrixScalarBuiltinSPInstruction extends BuiltinBinarySPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public MatrixScalarBuiltinSPInstruction(Operator op,
											CPOperand in1,
											CPOperand in2,
											CPOperand out,
											String opcode,
											String instr){
		super(op, in1, in2, out, 2, opcode, instr);
	}
	
	@Override 
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{	
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		try {
		    PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("myfile.txt", true)));
		    out.println("MatrixScalarBuiltinSPInstruction:" + getOpcode());
		    out.close();
		} catch (IOException e) {
		    //exception handling left as an exercise for the reader
		}
		
		CPOperand mat = ( input1.getDataType() == DataType.MATRIX ) ? input1 : input2;
		CPOperand scalar = ( input1.getDataType() == DataType.MATRIX ) ? input2 : input1;
		
		ScalarObject constant = (ScalarObject) sec.getScalarInput(scalar.getName(), scalar.getValueType(), scalar.isLiteral());
		
		//get input
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockedRDDHandleForVariable( mat.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1.mapToPair(new RDDMatrixScalarBuiltinUnaryOp( (ScalarOperator)_optr, constant.getDoubleValue()));
		
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		if(!mcOut.dimsKnown()) {
			throw new DMLRuntimeException("The output dimensions are not specified for MatrixScalarBuiltinSPInstruction");
		}
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), mat.getName());
	}
	
	public static class RDDMatrixScalarBuiltinUnaryOp implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock> {
		private static final long serialVersionUID = -3128192099832877491L;
		ScalarOperator sc_op;
		
		public RDDMatrixScalarBuiltinUnaryOp(ScalarOperator sc_op, double constant) {	
			this.sc_op = sc_op;
			this.sc_op.setConstant(constant);
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
			MatrixBlock resultBlock = (MatrixBlock) kv._2.scalarOperations(sc_op, new MatrixBlock());
			return new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, resultBlock);
		}
		
	}
}
