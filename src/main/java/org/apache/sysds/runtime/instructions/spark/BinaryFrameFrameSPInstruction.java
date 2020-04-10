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
import org.apache.spark.broadcast.Broadcast;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class BinaryFrameFrameSPInstruction extends BinarySPInstruction {
	protected BinaryFrameFrameSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(SPType.Binary, op, in1, in2, out, opcode, istr);
	}

	public static BinarySPInstruction parseInstruction ( String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields (parts, 3);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[3]);
		Types.DataType dt1 = in1.getDataType();
		Types.DataType dt2 = in2.getDataType();
		Operator operator = InstructionUtils.parseBinaryOrBuiltinOperator(opcode, in1, in2);
		if(dt1 == Types.DataType.FRAME && dt2 == Types.DataType.FRAME)
			return new BinaryFrameFrameSPInstruction(operator, in1, in2, out, opcode, str);
		else
			throw new DMLRuntimeException("Frame binary operation not yet implemented for frame-scalar, or frame-matrix");
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		// Get input RDDs
		JavaPairRDD<Long, FrameBlock> in1 = sec.getFrameBinaryBlockRDDHandleForVariable(input1.getName());
		// get schema frame-block
		Broadcast<FrameBlock> fb = sec.getSparkContext().broadcast(sec.getFrameInput(input2.getName()));
		JavaPairRDD<Long, FrameBlock> out = in1.mapValues(new isCorrectbySchema(fb.getValue()));
		//release input frame
		sec.releaseFrameInput(input2.getName());
		//set output RDD
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
	}

	private static class isCorrectbySchema implements Function<FrameBlock,FrameBlock> {
		private static final long serialVersionUID = 5850400295183766400L;

		private FrameBlock schema_frame = null;

		public isCorrectbySchema(FrameBlock fb_name ) {
			schema_frame = fb_name;
		}

		@Override
		public FrameBlock call(FrameBlock arg0) throws Exception {
			return arg0.dropInvalid(schema_frame);
		}
	}
}
