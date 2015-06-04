package com.ibm.bi.dml.runtime.instructions.spark;

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
import com.ibm.bi.dml.runtime.instructions.spark.functions.SparkUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;

public class ScalarMatrixRelationalSPInstruction extends RelationalBinarySPInstruction  {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public ScalarMatrixRelationalSPInstruction(Operator op, 
			   								   CPOperand in1, 
			   								   CPOperand in2, 
			   								   CPOperand out, 
			   								   String opcode,
			   								   String istr){
		super(op, in1, in2, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{	
		String opcode = getOpcode();
		if ( opcode.equalsIgnoreCase("==") || opcode.equalsIgnoreCase("!=") || opcode.equalsIgnoreCase("<")
			|| opcode.equalsIgnoreCase(">") || opcode.equalsIgnoreCase("<=") || opcode.equalsIgnoreCase(">=")) {
			
			SparkExecutionContext sec = (SparkExecutionContext)ec;
	
			// Get input RDD
			String rddVar 	= 	(input1.getDataType() == DataType.MATRIX) ? input1.getName() : input2.getName();
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( rddVar );
			
			// Get operator and scalar
			CPOperand scalar = ( input1.getDataType() == DataType.MATRIX ) ? input2 : input1;
			ScalarObject constant = (ScalarObject) ec.getScalarInput(scalar.getName(), scalar.getValueType(), scalar.isLiteral());
			ScalarOperator sc_op = (ScalarOperator) _optr;
			sc_op.setConstant(constant.getDoubleValue());
			
			//execute scalar matrix arithmetic instruction
			MatrixCharacteristics mc = sec.getMatrixCharacteristics(rddVar);
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1.mapToPair( new RDDScalarMatrixRelationalFunction(sc_op) );
			
			//put output RDD handle into symbol table
			MatrixCharacteristics mcOut = ec.getMatrixCharacteristics(output.getName());
			if(!mcOut.dimsKnown()) {
				if(!mc.dimsKnown())
					throw new DMLRuntimeException("The output dimensions are not specified for ScalarMatrixRelationalSPInstruction");
				else
					sec.getMatrixCharacteristics(output.getName()).set(mc);
			}
			
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), rddVar);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode in Instruction: " + toString());
		}
	}
	
	private static class RDDScalarMatrixRelationalFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 8197406787010296291L;

		private ScalarOperator sc_op = null;
		
		public RDDScalarMatrixRelationalFunction(ScalarOperator sc_op) {
			this.sc_op = sc_op;
		}
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) 
			throws Exception 
		{
			MatrixBlock resultBlk = (MatrixBlock) arg0._2.scalarOperations(sc_op, new MatrixBlock());	
			return new Tuple2<MatrixIndexes, MatrixBlock>(arg0._1, resultBlk);
		}
		
	}
}
