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
import org.apache.sysds.runtime.functionobjects.COV;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.COVOperator;
import scala.Tuple2;

public class CovarianceSPInstruction extends BinarySPInstruction {

	private CovarianceSPInstruction(COVOperator op, CPOperand in, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(SPType.Covariance, op, in, in2, out, opcode, istr);
	}

	private CovarianceSPInstruction(COVOperator op, CPOperand in, CPOperand in2, CPOperand in3, CPOperand out, String opcode, String istr) {
		super(SPType.Covariance, op, in, in2, out, opcode, istr);
	}

	public static CovarianceSPInstruction parseInstruction(String str)
	{
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in3 = null;
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if( !opcode.equalsIgnoreCase(Opcodes.COV.toString()) ) {
			throw new DMLRuntimeException("CovarianceCPInstruction.parseInstruction():: Unknown opcode " + opcode);
		}
		
		COVOperator cov = new COVOperator(COV.getCOMFnObject());
		if ( parts.length == 4 ) {
			// CP.cov.mVar0.mVar1.mVar2
			parseBinaryInstruction(str, in1, in2, out);
			return new CovarianceSPInstruction(cov, in1, in2, out, opcode, str);
		} else if ( parts.length == 5 ) {
			// CP.cov.mVar0.mVar1.mVar2.mVar3
			in3 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			parseBinaryInstruction(str, in1, in2, in3, out);
			return new CovarianceSPInstruction(cov, in1, in2, in3, out, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Invalid number of arguments in Instruction: " + str);
		}
	}
	
	@Override
	public void processInstruction( ExecutionContext ec ) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		COVOperator cop = ((COVOperator)_optr); 
		
		//get input
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryMatrixBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryMatrixBlockRDDHandleForVariable( input2.getName() );
		
		//process central moment instruction
		CM_COV_Object cmobj = null; 
		if( input3 == null ) { //w/o weights
			cmobj = in1.join( in2 )
				.values().map(new RDDCOVFunction(cop))
				.fold(new CM_COV_Object(), new RDDCOVReduceFunction(cop));
		}
		else { //with weights
			JavaPairRDD<MatrixIndexes,MatrixBlock> in3 = sec.getBinaryMatrixBlockRDDHandleForVariable( input3.getName() );
			cmobj = in1.join( in2 ).join( in3 )
				.values().map(new RDDCOVWeightsFunction(cop))
				.fold(new CM_COV_Object(), new RDDCOVReduceFunction(cop));
		}

		//create scalar output (no lineage information required)
		double val = cmobj.getRequiredResult(_optr);
		ec.setScalarOutput(output.getName(), new DoubleObject(val));
	}

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

	private static class RDDCOVReduceFunction implements Function2<CM_COV_Object, CM_COV_Object, CM_COV_Object>
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
			//execute cov combine operations
			_op.fn.execute(arg0, arg1);			
			return arg0;
		}
	}
}
