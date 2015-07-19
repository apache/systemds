/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;

import scala.Tuple2;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.COV;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CM_COV_Object;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.cp.CovarianceCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.DoubleObject;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.COVOperator;

/**
 * 
 */
public class CovarianceSPInstruction extends BinarySPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public CovarianceSPInstruction(AggregateUnaryOperator op, CPOperand in, CPOperand in2, CPOperand out, 
			                      String opcode, String istr)
	{
		super(op, in, in2, out, opcode, istr);
	}
	
	public CovarianceSPInstruction(AggregateUnaryOperator op, CPOperand in, CPOperand in2, CPOperand in3, CPOperand out, 
            String opcode, String istr)
	{
		super(op, in, in2, out, opcode, istr);
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction(String str)
		throws DMLRuntimeException 
	{
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in3 = null;
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if( !opcode.equalsIgnoreCase("cov") ) {
			throw new DMLRuntimeException("CovarianceCPInstruction.parseInstruction():: Unknown opcode " + opcode);
		}
		
		COVOperator cov = new COVOperator(COV.getCOMFnObject());
		if ( parts.length == 4 ) {
			// CP.cov.mVar0.mVar1.mVar2
			parseBinaryInstruction(str, in1, in2, out);
			return new CovarianceCPInstruction(cov, in1, in2, out, opcode, str);
		} else if ( parts.length == 5 ) {
			// CP.cov.mVar0.mVar1.mVar2.mVar3
			in3 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			parseBinaryInstruction(str, in1, in2, in3, out);
			return new CovarianceCPInstruction(cov, in1, in2, in3, out, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Invalid number of arguments in Instruction: " + str);
		}
	}
	
	@Override
	public void processInstruction( ExecutionContext ec )
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		COVOperator cop = ((COVOperator)_optr); 
		
		//get input
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryBlockRDDHandleForVariable( input2.getName() );
				
		//process central moment instruction
		CM_COV_Object cmobj = null; 
		if( input3 == null ) //w/o weights
		{
			cmobj = in1.join( in2 )
					   .values().map(new RDDCOVFunction(cop))
			           .reduce(new RDDCOVReduceFunction(cop));
		}
		else //with weights
		{
			JavaPairRDD<MatrixIndexes,MatrixBlock> in3 = sec.getBinaryBlockRDDHandleForVariable( input3.getName() );
			cmobj = in1.join( in2 )
					   .join( in3 )
					   .values().map(new RDDCOVWeightsFunction(cop))
			           .reduce(new RDDCOVReduceFunction(cop));
		}

		//create scalar output (no lineage information required)
		double val = cmobj.getRequiredResult(_optr);
		DoubleObject ret = new DoubleObject(output.getName(), val);
		ec.setScalarOutput(output.getName(), ret);
	}

	/**
	 * 
	 */
	private static class RDDCOVFunction implements Function<Tuple2<MatrixBlock,MatrixBlock>, CM_COV_Object>
	{
		private static final long serialVersionUID = -9088449969750217519L;
		
		private COVOperator _op = null;
		
		public RDDCOVFunction( COVOperator op ) {
			_op = op;
		}

		@Override
		public CM_COV_Object call(Tuple2<MatrixBlock,MatrixBlock> arg0) 
			throws Exception 
		{
			MatrixBlock input1 = arg0._1();
			MatrixBlock input2 = arg0._2();
			
			//execute cov operations
			return input1.covOperations(_op, input2);
		}
	}
	
	/**
	 * 
	 */
	private static class RDDCOVWeightsFunction implements Function<Tuple2<Tuple2<MatrixBlock,MatrixBlock>,MatrixBlock>, CM_COV_Object> 
	{
		private static final long serialVersionUID = 1945166819152577077L;
		
		private COVOperator _op = null;
		
		public RDDCOVWeightsFunction( COVOperator op ) {
			_op = op;
		}

		@Override
		public CM_COV_Object call(Tuple2<Tuple2<MatrixBlock,MatrixBlock>,MatrixBlock> arg0) 
			throws Exception 
		{
			MatrixBlock input1 = arg0._1()._1();
			MatrixBlock input2 = arg0._1()._2();
			MatrixBlock weights = arg0._2();
			
			//execute cov operations
			return input1.covOperations(_op, input2, weights);
		}
	}
	
	/**
	 * 
	 */
	public class RDDCOVReduceFunction implements Function2<CM_COV_Object, CM_COV_Object, CM_COV_Object>
	{
		private static final long serialVersionUID = 1118102911706607118L;
		
		private COVOperator _op = null;
		
		public RDDCOVReduceFunction( COVOperator op ) {
			_op = op;
		}

		@Override
		public CM_COV_Object call(CM_COV_Object arg0, CM_COV_Object arg1) 
			throws Exception 
		{
			CM_COV_Object out = new CM_COV_Object();
			
			//execute cov combine operations
			_op.fn.execute(out, arg0);
			_op.fn.execute(out, arg1);
			
			return out;
		}
	}
}
