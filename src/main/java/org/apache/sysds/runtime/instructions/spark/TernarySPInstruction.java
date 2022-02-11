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
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.TernaryOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import scala.Tuple2;

import java.io.Serializable;

public class TernarySPInstruction extends ComputationSPInstruction {
	protected TernarySPInstruction(TernaryOperator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String opcode, String str) {
		super(SPType.Ternary, op, in1, in2, in3, out, opcode, str);
	}

	public static TernarySPInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode=parts[0];
		CPOperand operand1 = new CPOperand(parts[1]);
		CPOperand operand2 = new CPOperand(parts[2]);
		CPOperand operand3 = new CPOperand(parts[3]);
		CPOperand outOperand = new CPOperand(parts[4]);
		TernaryOperator op = InstructionUtils.parseTernaryOperator(opcode);
		if(operand1.isFrame() && operand2.isScalar() && opcode.contains("map"))
			return  new TernaryFrameScalarSPInstruction(op, operand1, operand2, operand3, outOperand, opcode, str);
		else
		 return new TernarySPInstruction(op, operand1, operand2, operand3, outOperand, opcode,str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = !input1.isMatrix() ? null :
			sec.getBinaryMatrixBlockRDDHandleForVariable(input1.getName());
		JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = !input2.isMatrix() ? null :
			sec.getBinaryMatrixBlockRDDHandleForVariable(input2.getName());
		JavaPairRDD<MatrixIndexes,MatrixBlock> in3 = !input3.isMatrix() ? null :
			sec.getBinaryMatrixBlockRDDHandleForVariable(input3.getName());
		MatrixBlock m1 = input1.isMatrix() ? null :
			new MatrixBlock(ec.getScalarInput(input1).getDoubleValue());
		MatrixBlock m2 = input2.isMatrix() ? null :
			new MatrixBlock(ec.getScalarInput(input2).getDoubleValue());
		MatrixBlock m3 = input3.isMatrix() ? null :
			new MatrixBlock(ec.getScalarInput(input3).getDoubleValue());
		
		TernaryOperator op = (TernaryOperator) _optr;
		
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		if( input1.isMatrix() && !input2.isMatrix() && !input3.isMatrix() )
			out = in1.mapValues(new TernaryFunctionMSS(op, m1, m2, m3));
		else if( !input1.isMatrix() && input2.isMatrix() && !input3.isMatrix() )
			out = in2.mapValues(new TernaryFunctionSMS(op, m1, m2, m3));
		else if( !input1.isMatrix() && !input2.isMatrix() && input3.isMatrix() )
			out = in3.mapValues(new TernaryFunctionSSM(op, m1, m2, m3));
		else if( input1.isMatrix() && input2.isMatrix() && !input3.isMatrix() )
			out = in1.join(in2).mapValues(new TernaryFunctionMMS(op, m1, m2, m3));
		else if( input1.isMatrix() && !input2.isMatrix() && input3.isMatrix() )
			out = in1.join(in3).mapValues(new TernaryFunctionMSM(op, m1, m2, m3));
		else if( !input1.isMatrix() && input2.isMatrix() && input3.isMatrix() )
			out = in2.join(in3).mapValues(new TernaryFunctionSMM(op, m1, m2, m3));
		else // all matrices
			out = in1.join(in2).join(in3).mapValues(new TernaryFunctionMMM(op, m1, m2, m3));
		
		//set output RDD
		updateTernaryOutputDataCharacteristics(sec);
		sec.setRDDHandleForVariable(output.getName(), out);
		if( input1.isMatrix() )
			sec.addLineageRDD(output.getName(), input1.getName());
		if( input2.isMatrix() )
			sec.addLineageRDD(output.getName(), input2.getName());
		if( input3.isMatrix() )
			sec.addLineageRDD(output.getName(), input3.getName());
	}
	
	protected void updateTernaryOutputDataCharacteristics(SparkExecutionContext sec) {
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
		for(CPOperand input : new CPOperand[]{input1, input2, input3})
			if( input.isMatrix() ) {
				DataCharacteristics mc = sec.getDataCharacteristics(input.getName());
				if( mc.dimsKnown() )
					mcOut.set(mc);
			}
	}
	
	private static abstract class TernaryFunction implements Serializable {
		private static final long serialVersionUID = 8345737737972434426L;
		protected final TernaryOperator _op;
		protected final MatrixBlock _m1, _m2, _m3;
		public TernaryFunction(TernaryOperator op, MatrixBlock m1, MatrixBlock m2, MatrixBlock m3) {
			_op = op; _m1 = m1; _m2 = m2; _m3 = m3;
		}
	}
	
	private static class TernaryFunctionMSS extends TernaryFunction implements Function<MatrixBlock, MatrixBlock> {
		private static final long serialVersionUID = 1L;
		public TernaryFunctionMSS(TernaryOperator op, MatrixBlock m1, MatrixBlock m2, MatrixBlock m3) {
			super(op, m1, m2, m3);
		}
		@Override
		public MatrixBlock call(MatrixBlock v1) throws Exception {
			return v1.ternaryOperations(_op, _m2, _m3, new MatrixBlock());
		}
	}
	
	private static class TernaryFunctionSMS extends TernaryFunction implements Function<MatrixBlock, MatrixBlock> {
		private static final long serialVersionUID = 1L;
		public TernaryFunctionSMS(TernaryOperator op, MatrixBlock m1, MatrixBlock m2, MatrixBlock m3) {
			super(op, m1, m2, m3);
		}
		@Override
		public MatrixBlock call(MatrixBlock v1) throws Exception {
			return _m1.ternaryOperations(_op, v1, _m3, new MatrixBlock());
		}
	}
	
	private static class TernaryFunctionSSM extends TernaryFunction implements Function<MatrixBlock, MatrixBlock> {
		private static final long serialVersionUID = 1L;
		public TernaryFunctionSSM(TernaryOperator op, MatrixBlock m1, MatrixBlock m2, MatrixBlock m3) {
			super(op, m1, m2, m3);
		}
		@Override
		public MatrixBlock call(MatrixBlock v1) throws Exception {
			return _m1.ternaryOperations(_op, _m2, v1, new MatrixBlock());
		}
	}
	
	private static class TernaryFunctionMMS extends TernaryFunction implements Function<Tuple2<MatrixBlock,MatrixBlock>, MatrixBlock> {
		private static final long serialVersionUID = 1L;
		public TernaryFunctionMMS(TernaryOperator op, MatrixBlock m1, MatrixBlock m2, MatrixBlock m3) {
			super(op, m1, m2, m3);
		}
		@Override
		public MatrixBlock call(Tuple2<MatrixBlock, MatrixBlock> v1) throws Exception {
			return v1._1().ternaryOperations(_op, v1._2(), _m3, new MatrixBlock());
		}
	}
	
	private static class TernaryFunctionMSM extends TernaryFunction implements Function<Tuple2<MatrixBlock,MatrixBlock>, MatrixBlock> {
		private static final long serialVersionUID = 1L;
		public TernaryFunctionMSM(TernaryOperator op, MatrixBlock m1, MatrixBlock m2, MatrixBlock m3) {
			super(op, m1, m2, m3);
		}
		@Override
		public MatrixBlock call(Tuple2<MatrixBlock, MatrixBlock> v1) throws Exception {
			return v1._1().ternaryOperations(_op, _m2, v1._2(), new MatrixBlock());
		}
	}
	
	private static class TernaryFunctionSMM extends TernaryFunction implements Function<Tuple2<MatrixBlock,MatrixBlock>, MatrixBlock> {
		private static final long serialVersionUID = 1L;
		public TernaryFunctionSMM(TernaryOperator op, MatrixBlock m1, MatrixBlock m2, MatrixBlock m3) {
			super(op, m1, m2, m3);
		}
		@Override
		public MatrixBlock call(Tuple2<MatrixBlock, MatrixBlock> v1) throws Exception {
			return _m1.ternaryOperations(_op, v1._1(), v1._2(), new MatrixBlock());
		}
	}
	
	private static class TernaryFunctionMMM extends TernaryFunction implements Function<Tuple2<Tuple2<MatrixBlock,MatrixBlock>,MatrixBlock>, MatrixBlock> {
		private static final long serialVersionUID = 1L;
		public TernaryFunctionMMM(TernaryOperator op, MatrixBlock m1, MatrixBlock m2, MatrixBlock m3) {
			super(op, m1, m2, m3);
		}
		@Override
		public MatrixBlock call(Tuple2<Tuple2<MatrixBlock,MatrixBlock>,MatrixBlock> v1) throws Exception {
			return v1._1()._1().ternaryOperations(_op, v1._1()._2(), v1._2(), new MatrixBlock());
		}
	}
}
