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
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.Operator;
import scala.Tuple2;

public class UnaryFrameSPInstruction extends UnarySPInstruction {
	protected UnaryFrameSPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String instr) {
		super(SPInstruction.SPType.Unary, op, in, out, opcode, instr);
	}

	public static UnaryFrameSPInstruction parseInstruction(String str) {
		CPOperand in = new CPOperand("", Types.ValueType.UNKNOWN, Types.DataType.UNKNOWN);
		CPOperand out = new CPOperand("", Types.ValueType.UNKNOWN, Types.DataType.UNKNOWN);
		String opcode = parseUnaryInstruction(str, in, out);
		return new UnaryFrameSPInstruction(InstructionUtils.parseUnaryOperator(opcode), in, out, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext) ec;
		if(getOpcode().equals(OpOp1.DETECTSCHEMA.toString()))
			detectSchema(sec);
		else if(getOpcode().equals(OpOp1.COLNAMES.toString()))
			columnNames(sec);
		else
			throw new DMLScriptException("Opcode '" + getOpcode() + "' is not a valid UnaryFrameSPInstruction");
	}

	private void columnNames(SparkExecutionContext sec) {
		// get input
		JavaPairRDD<Long, FrameBlock> in = sec.getFrameBinaryBlockRDDHandleForVariable(input1.getName());
		// get the first row block (frames are only blocked rowwise) and get its column names
		FrameBlock outFrame = in.lookup(1L).get(0).getColumnNamesAsFrame();
		sec.setFrameOutput(output.getName(), outFrame);
	}

	public void detectSchema(SparkExecutionContext sec) {
		// get input
		JavaPairRDD<Long, FrameBlock> in = sec.getFrameBinaryBlockRDDHandleForVariable(input1.getName());
		JavaPairRDD<Long, FrameBlock> out = in.mapToPair(new DetectSchemaUsingRows());
		FrameBlock outFrame = out.values().reduce(new MergeFrame());
		sec.setFrameOutput(output.getName(), outFrame);
	}

	private static class DetectSchemaUsingRows implements PairFunction<Tuple2<Long, FrameBlock>, Long, FrameBlock> {
		private static final long serialVersionUID = 5850400295183766400L;

		@Override
		public Tuple2<Long, FrameBlock> call(Tuple2<Long, FrameBlock> arg0) throws Exception {
			FrameBlock resultBlock = new FrameBlock(arg0._2.detectSchemaFromRow(Lop.SAMPLE_FRACTION));
			return new Tuple2<>(1L, resultBlock);
		}
	}

	private static class MergeFrame implements Function2<FrameBlock, FrameBlock, FrameBlock> {
		private static final long serialVersionUID = 942744896521069893L;

		@Override
		public FrameBlock call(FrameBlock arg0, FrameBlock arg1) throws Exception {
			return new FrameBlock(FrameBlock.mergeSchema(arg0, arg1));
		}
	}
}
