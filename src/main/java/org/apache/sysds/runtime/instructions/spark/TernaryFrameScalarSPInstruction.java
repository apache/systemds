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
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.operators.TernaryOperator;

public class TernaryFrameScalarSPInstruction extends TernarySPInstruction {
	protected TernaryFrameScalarSPInstruction(TernaryOperator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
			String opcode, String istr) {
		super(op, in1, in2, in3, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;

		// Get input RDDs
		JavaPairRDD<Long, FrameBlock> in1 = sec.getFrameBinaryBlockRDDHandleForVariable(input1.getName());
		String expression = sec.getScalarInput(input2).getStringValue();
		long margin = ec.getScalarInput(input3).getLongValue();

		// Create local compiled functions (once) and execute on RDD
		JavaPairRDD<Long, FrameBlock> out = in1.mapValues(new RDDStringProcessing(expression, margin));

		if(expression.contains("jaccardSim")) {
			long rows = sec.getDataCharacteristics(output.getName()).getRows();
			sec.getDataCharacteristics(output.getName()).setDimension(rows, rows);
		} else {
			long rows = sec.getDataCharacteristics(output.getName()).getRows();
			long cols = sec.getDataCharacteristics(output.getName()).getCols();
			sec.getDataCharacteristics(output.getName()).setDimension(rows, cols);
		}
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
	}

	private static class RDDStringProcessing implements Function<FrameBlock,FrameBlock> {
		private static final long serialVersionUID = 5850400295183766400L;

		private String _expr = null;
		private long _margin = -1;

		public RDDStringProcessing(String expr, long margin) {
			_expr = expr;
			_margin = margin;
		}

		@Override
		public FrameBlock call(FrameBlock arg0) throws Exception {
			FrameBlock fb =  arg0.map(_expr, _margin);
			return fb;
		}
	}
}
