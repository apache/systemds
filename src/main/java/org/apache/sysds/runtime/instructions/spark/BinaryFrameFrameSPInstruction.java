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
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import scala.Tuple2;

public class BinaryFrameFrameSPInstruction extends BinarySPInstruction {
	protected BinaryFrameFrameSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out,
			String opcode, String istr) {
		super(SPType.Binary, op, in1, in2, out, opcode, istr);
	}


	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;

		// Get input RDDs
		JavaPairRDD<Long, FrameBlock> in1 = sec.getFrameBinaryBlockRDDHandleForVariable(input1.getName());
		JavaPairRDD<Long, FrameBlock> out = null;
		
		if(getOpcode().equals("dropInvalidType")) {
			// get schema frame-block
			Broadcast<FrameBlock> fb = sec.getSparkContext().broadcast(sec.getFrameInput(input2.getName()));
			out = in1.mapValues(new isCorrectbySchema(fb.getValue()));
			//release input frame
			sec.releaseFrameInput(input2.getName());
		}
		else if(getOpcode().equals("valueSwap")) {
			// Perform computation using input frames, and produce the result frame
			Broadcast<FrameBlock> fb = sec.getSparkContext().broadcast(sec.getFrameInput(input2.getName()));
			out = in1.mapValues(new valueSwapBySchema(fb.getValue()));
			// Attach result frame with FrameBlock associated with output_name
			sec.releaseFrameInput(input2.getName());
		}
		else if(getOpcode().equals("applySchema")){
			Broadcast<FrameBlock> fb = sec.getSparkContext().broadcast(sec.getFrameInput(input2.getName()));
			out = in1.mapValues(new applySchema(fb.getValue()));
			sec.releaseFrameInput(input2.getName());
		}
		else {
			JavaPairRDD<Long, FrameBlock> in2 = sec.getFrameBinaryBlockRDDHandleForVariable(input2.getName());
			// create output frame
			BinaryOperator dop = (BinaryOperator) _optr;
			// check for binary operations
			out = in1.join(in2).mapValues(new FrameComparison(dop));
		}
		
		//set output RDD and maintain dependencies
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		if(!getOpcode().equals("dropInvalidType") && //
			!getOpcode().equals("valueSwap") && //
			!getOpcode().equals("applySchema"))
			sec.addLineageRDD(output.getName(), input2.getName());
	}

	private static class isCorrectbySchema implements Function<FrameBlock,FrameBlock> {
		private static final long serialVersionUID = 5850400295183766400L;

		private FrameBlock schema_frame;

		public isCorrectbySchema(FrameBlock fb_name ) {
			schema_frame = fb_name;
		}

		@Override
		public FrameBlock call(FrameBlock arg0) throws Exception {
			return arg0.dropInvalidType(schema_frame);
		}
	}

	private static class FrameComparison implements Function<Tuple2<FrameBlock, FrameBlock>, FrameBlock> {
		private static final long serialVersionUID = 5850400295183766401L;
		private final BinaryOperator bop;
		public FrameComparison(BinaryOperator op){
			bop = op;
		}

		@Override
		public FrameBlock call(Tuple2<FrameBlock, FrameBlock> arg0) throws Exception {
			return arg0._1().binaryOperations(bop, arg0._2(), null);
		}
	}

	private static class valueSwapBySchema implements Function<FrameBlock,FrameBlock> {
		private static final long serialVersionUID = 5850400295183766402L;

		private FrameBlock schema_frame;

		public valueSwapBySchema(FrameBlock fb_name ) {
			schema_frame = fb_name;
		}

		@Override
		public FrameBlock call(FrameBlock arg0) throws Exception {
			return arg0.valueSwap(schema_frame);
		}
	}


	private static class applySchema implements Function<FrameBlock, FrameBlock>{
		private static final long serialVersionUID = 58504021316402L;

		private FrameBlock schema;

		public applySchema(FrameBlock schema ) {
			this.schema = schema;
		}

		@Override
		public FrameBlock call(FrameBlock arg0) throws Exception {
			return arg0.applySchema(schema);
		}
	}
}
