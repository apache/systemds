/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.functionobjects.CM;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator.AggregateOperationTypes;
import scala.Tuple2;

public class CentralMomentSPInstruction extends UnarySPInstruction {

	private CentralMomentSPInstruction(CMOperator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String opcode, String str) {
		super(SPType.CentralMoment, op, in1, in2, in3, out, opcode, str);
	}

	public static CentralMomentSPInstruction parseInstruction(String str) {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = null; 
		CPOperand in3 = null; 
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0]; 
		
		//check supported opcode
		if( !opcode.equalsIgnoreCase(Opcodes.CM.toString()) ) {
			throw new DMLRuntimeException("Unsupported opcode "+opcode);
		}
			
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
		return new CentralMomentSPInstruction(cm, in1, in2, in3, out, opcode, str);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec ) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//parse 'order' input argument 
		CPOperand scalarInput = (input3==null ? input2 : input3);
		ScalarObject order = ec.getScalarInput(scalarInput); 
		CMOperator cop = ((CMOperator)_optr); 
		if ( cop.getAggOpType() == AggregateOperationTypes.INVALID ) {
			cop.setCMAggOp((int)order.getLongValue());
		}
		
		//get input
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryMatrixBlockRDDHandleForVariable( input1.getName() );
				
		//process central moment instruction
		CmCovObject cmobj = null; 
		if( input3 == null ) //w/o weights
		{
			cmobj = in1.values().map(new RDDCMFunction(cop))
			           .fold(new CmCovObject(), new RDDCMReduceFunction(cop));
		}
		else //with weights
		{
			JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryMatrixBlockRDDHandleForVariable( input2.getName() );
			cmobj = in1.join( in2 )
					   .values().map(new RDDCMWeightsFunction(cop))
			           .fold(new CmCovObject(), new RDDCMReduceFunction(cop));
		}

		//create scalar output (no lineage information required)
		double val = cmobj.getRequiredResult(_optr);
		ec.setScalarOutput(output.getName(), new DoubleObject(val));
	}

	private static class RDDCMFunction implements Function<MatrixBlock, CmCovObject> 
	{
		private static final long serialVersionUID = 2293839116041610644L;
		
		private CMOperator _op = null;
		
		public RDDCMFunction( CMOperator op ) {
			_op = op;
		}

		@Override
		public CmCovObject call(MatrixBlock arg0) 
			throws Exception 
		{
			//execute cm operations
			return arg0.cmOperations(_op);
		}
	}

	private static class RDDCMWeightsFunction implements Function<Tuple2<MatrixBlock,MatrixBlock>, CmCovObject> 
	{
		private static final long serialVersionUID = -8949715516574052497L;
		
		private CMOperator _op = null;
		
		public RDDCMWeightsFunction( CMOperator op ) {
			_op = op;
		}

		@Override
		public CmCovObject call(Tuple2<MatrixBlock,MatrixBlock> arg0) 
			throws Exception 
		{
			MatrixBlock input = arg0._1();
			MatrixBlock weights = arg0._2();
			
			//execute cm operations
			return input.cmOperations(_op, weights);
		}
	}
	
	private static class RDDCMReduceFunction implements Function2<CmCovObject, CmCovObject, CmCovObject>
	{

		private static final long serialVersionUID = 3272260751983866544L;
		
		private CMOperator _op = null;
		
		public RDDCMReduceFunction( CMOperator op ) {
			_op = op;
		}

		@Override
		public CmCovObject call(CmCovObject arg0, CmCovObject arg1) 
			throws Exception 
		{
			//execute cm combine operations
			_op.fn.execute(arg0, arg1);		
			return arg0;
		}
	}
}
