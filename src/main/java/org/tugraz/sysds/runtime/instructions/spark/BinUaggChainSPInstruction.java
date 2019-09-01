/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.BinaryOperator;

public class BinUaggChainSPInstruction extends UnarySPInstruction {
	// operators
	private BinaryOperator _bOp = null;
	private AggregateUnaryOperator _uaggOp = null;

	private BinUaggChainSPInstruction(CPOperand in, CPOperand out, BinaryOperator bop, AggregateUnaryOperator uaggop,
			String opcode, String istr) {
		super(SPType.BinUaggChain, null, in, out, opcode, istr);
		_bOp = bop;
		_uaggOp = uaggop;

	}

	public static BinUaggChainSPInstruction parseInstruction ( String str ) {
		//parse instruction parts (without exec type)
		String[] parts = InstructionUtils.getInstructionPartsWithValueType( str );
		InstructionUtils.checkNumFields( parts, 4 );
		String opcode = parts[0];
		BinaryOperator bop = InstructionUtils.parseBinaryOperator(parts[1]);
		AggregateUnaryOperator uaggop = InstructionUtils.parseBasicAggregateUnaryOperator(parts[2]);
		CPOperand in = new CPOperand(parts[3]);
		CPOperand out = new CPOperand(parts[4]);
		return new BinUaggChainSPInstruction(in, out, bop, uaggop, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//get input
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryMatrixBlockRDDHandleForVariable( input1.getName() );
		
		//execute unary builtin operation
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = 
				in.mapValues(new RDDBinUaggChainFunction(_bOp, _uaggOp));
		
		//set output RDD
		updateUnaryOutputDataCharacteristics(sec);
		sec.setRDDHandleForVariable(output.getName(), out);	
		sec.addLineageRDD(output.getName(), input1.getName());
	}

	public static class RDDBinUaggChainFunction implements Function<MatrixBlock,MatrixBlock> 
	{
		private static final long serialVersionUID = 886065328623752520L;
		
		private BinaryOperator _bOp = null;
		private AggregateUnaryOperator _uaggOp = null;
		
		public RDDBinUaggChainFunction(BinaryOperator bop, AggregateUnaryOperator uaggop) {
			_bOp = bop;
			_uaggOp = uaggop;
		}

		@Override
		public MatrixBlock call(MatrixBlock arg0) 
			throws Exception 
		{
			int blen = arg0.getNumRows();
			
			//perform unary aggregate operation
			MatrixBlock out1 = new MatrixBlock();
			arg0.aggregateUnaryOperations(_uaggOp, out1, blen, null);
			
			//strip-off correction
			out1.dropLastRowsOrColumns(_uaggOp.aggOp.correctionLocation);
		
			//perform binary operation
			MatrixBlock out2 = new MatrixBlock();
			return (MatrixBlock) arg0.binaryOperations(_bOp, out1, out2);
		}
	}
}
