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

package org.apache.sysml.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;

import scala.Tuple2;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.functionobjects.OffsetColumnIndex;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;

public class FrameAppendRSPInstruction extends AppendRSPInstruction
{
	public FrameAppendRSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, boolean cbind, String opcode, String istr)
	{
		super(op, in1, in2, out, cbind, opcode, istr);
	}
	
	public static FrameAppendRSPInstruction parseInstruction ( String str ) 
			throws DMLRuntimeException 
	{	
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields (parts, 4);
		
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[3]);
		boolean cbind = Boolean.parseBoolean(parts[4]);
		
		if(!opcode.equalsIgnoreCase("rappend"))
			throw new DMLRuntimeException("Unknown opcode while parsing a FrameAppendRSPInstruction: " + str);
		
		return new FrameAppendRSPInstruction(
				new ReorgOperator(OffsetColumnIndex.getOffsetColumnIndexFnObject(-1)), 
				in1, in2, out, cbind, opcode, str);
	}
		
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLRuntimeException 
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		JavaPairRDD<Long,FrameBlock> in1 = sec.getFrameBinaryBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<Long,FrameBlock> in2 = sec.getFrameBinaryBlockRDDHandleForVariable( input2.getName() );
		
		//execute reduce-append operations (partitioning preserving)
		JavaPairRDD<Long,FrameBlock> out = in1
				.join(in2)
				.mapValues(new ReduceSideAppendFunction(_cbind));
		
		//put output RDD handle into symbol table
		updateBinaryAppendOutputMatrixCharacteristics(sec, _cbind);
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		sec.addLineageRDD(output.getName(), input2.getName());		
	}
	
	/**
	 * 
	 */
	private static class ReduceSideAppendFunction implements Function<Tuple2<FrameBlock, FrameBlock>, FrameBlock> 
	{
		private static final long serialVersionUID = -97824903649667646L;

		private boolean _cbind = true;
				
		public ReduceSideAppendFunction(boolean cbind) {
			_cbind = cbind;
		}
		
		@Override
		public FrameBlock call(Tuple2<FrameBlock, FrameBlock> arg0)
			throws Exception 
		{
			FrameBlock left = arg0._1();
			FrameBlock right = arg0._2();
			
			return left.appendOperations(right, new FrameBlock(), _cbind);
		}
	}
}

