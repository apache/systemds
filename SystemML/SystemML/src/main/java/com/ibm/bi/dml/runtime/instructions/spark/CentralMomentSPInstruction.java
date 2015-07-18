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
import com.ibm.bi.dml.runtime.functionobjects.CM;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CM_COV_Object;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.cp.CentralMomentCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.cp.ScalarObject;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator.AggregateOperationTypes;

/**
 * 
 */
public class CentralMomentSPInstruction extends AggregateUnarySPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public CentralMomentSPInstruction(AggregateUnaryOperator op, CPOperand in, CPOperand out, String opcode, String istr){
		super(op, in, out, true, opcode, istr);
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
		CPOperand in2 = null; 
		CPOperand in3 = null; 
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		String opcode = InstructionUtils.getOpCode(str); 
		
		//check supported opcode
		if( !opcode.equalsIgnoreCase("cm") ) {
			throw new DMLRuntimeException("Unsupported opcode "+opcode);
		}
			
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		if ( parts.length == 4 ) {
			// Example: CP.cm.mVar0.Var1.mVar2; (without weights)
			in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			parseUnaryInstruction(str, in1, in2, out);
		}
		else if ( parts.length == 5) {
			// CP.cm.mVar0.mVar1.Var2.mVar3; (with weights)
			in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			in3 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			parseUnaryInstruction(str, in1, in2, in3, out);
		}
	 
		// Exact order of the central moment MAY NOT be known at compilation time.
		// We first try to parse the second argument as an integer, and if we fail, 
		// we simply pass -1 so that getCMAggOpType() picks up AggregateOperationTypes.INVALID.
		// It must be updated at run time in processInstruction() method.
		
		int cmOrder;
		try {
			if ( in3 == null ) {
				cmOrder = Integer.parseInt(in2.getName());
			}
			else {
				cmOrder = Integer.parseInt(in3.getName());
			}
		} catch(NumberFormatException e) {
			cmOrder = -1; // unknown at compilation time
		}
		
		AggregateOperationTypes opType = CMOperator.getCMAggOpType(cmOrder);
		CMOperator cm = new CMOperator(CM.getCMFnObject(opType), opType);
		return new CentralMomentCPInstruction(cm, in1, in2, in3, out, opcode, str);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec )
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//parse 'order' input argument 
		CPOperand scalarInput = (input3==null ? input2 : input3);
		ScalarObject order = ec.getScalarInput(scalarInput.getName(), scalarInput.getValueType(), scalarInput.isLiteral()); 
		CMOperator cop = ((CMOperator)_optr); 
		if ( cop.getAggOpType() == AggregateOperationTypes.INVALID ) {
			cop.setCMAggOp((int)order.getLongValue());
		}
		
		//get input
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
				
		//process central moment instruction
		CM_COV_Object cmobj = null; 
		if( input3 == null ) //w/o weights
		{
			cmobj = in1.values().map(new RDDCMFunction(cop))
			           .reduce(new RDDCMReduceFunction(cop));
		}
		else //with weights
		{
			JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
			cmobj = in1.join( in2 )
					   .values().map(new RDDCMWeightsFunction(cop))
			           .reduce(new RDDCMReduceFunction(cop));
		}

		//create scalar output (no lineage information required)
		double val = cmobj.getRequiredResult(_optr);
		DoubleObject ret = new DoubleObject(output.getName(), val);
		ec.setScalarOutput(output.getName(), ret);
	}

	/**
	 * 
	 */
	private static class RDDCMFunction implements Function<MatrixBlock, CM_COV_Object> 
	{
		private static final long serialVersionUID = 2293839116041610644L;
		
		private CMOperator _op = null;
		
		public RDDCMFunction( CMOperator op ) {
			_op = op;
		}

		@Override
		public CM_COV_Object call(MatrixBlock arg0) 
			throws Exception 
		{
			//execute cm operations
			return arg0.cmOperations(_op);
		}
	}
	
	/**
	 * 
	 */
	private static class RDDCMWeightsFunction implements Function<Tuple2<MatrixBlock,MatrixBlock>, CM_COV_Object> 
	{
		private static final long serialVersionUID = -8949715516574052497L;
		
		private CMOperator _op = null;
		
		public RDDCMWeightsFunction( CMOperator op ) {
			_op = op;
		}

		@Override
		public CM_COV_Object call(Tuple2<MatrixBlock,MatrixBlock> arg0) 
			throws Exception 
		{
			MatrixBlock input = arg0._1();
			MatrixBlock weights = arg0._2();
			
			//execute cm operations
			return input.cmOperations(_op, weights);
		}
	}
	
	/**
	 * 
	 */
	private static class RDDCMReduceFunction implements Function2<CM_COV_Object, CM_COV_Object, CM_COV_Object> 
	{
		private static final long serialVersionUID = 6175110783088073856L;
		
		private CMOperator _op = null;
		
		public RDDCMReduceFunction( CMOperator op ) {
			_op = op;
		}

		@Override
		public CM_COV_Object call(CM_COV_Object arg0, CM_COV_Object arg1) 
			throws Exception 
		{
			CM_COV_Object out = new CM_COV_Object();
			
			//execute cm combine operations
			_op.fn.execute(out, arg0);
			_op.fn.execute(out, arg1);
			
			return out;
		}
	}
}
