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
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.utils.FrameRDDAggregateUtils;
import org.apache.sysds.runtime.matrix.operators.Operator;
import scala.Tuple2;

public class FrameAppendRSPInstruction extends AppendRSPInstruction {

	protected FrameAppendRSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, boolean cbind,
			String opcode, String istr) {
		super(op, in1, in2, out, cbind, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		JavaPairRDD<Long,FrameBlock> in1 = sec.getFrameBinaryBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<Long,FrameBlock> in2 = sec.getFrameBinaryBlockRDDHandleForVariable( input2.getName() );
		JavaPairRDD<Long,FrameBlock> out = null;
		long leftRows = sec.getDataCharacteristics(input1.getName()).getRows();
		
		if(_cbind) {
			JavaPairRDD<Long,FrameBlock> in1Aligned = in1.mapToPair(new ReduceSideAppendAlignFunction(leftRows));
			in1Aligned = FrameRDDAggregateUtils.mergeByKey(in1Aligned);
			JavaPairRDD<Long,FrameBlock> in2Aligned = in2.mapToPair(new ReduceSideAppendAlignFunction(leftRows));
			in2Aligned = FrameRDDAggregateUtils.mergeByKey(in2Aligned);
			
			out = in1Aligned.join(in2Aligned).mapValues(new ReduceSideColumnsFunction(_cbind));
		} else {	//rbind
			JavaPairRDD<Long,FrameBlock> right = in2.mapToPair( new ReduceSideAppendRowsFunction(leftRows));
			out = in1.union(right);
		}
		
		//put output RDD handle into symbol table
		updateBinaryAppendOutputDataCharacteristics(sec, _cbind);
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		sec.addLineageRDD(output.getName(), input2.getName());
		
		//update schema of output with merged input schemas
		sec.getFrameObject(output.getName()).setSchema(
			sec.getFrameObject(input1.getName()).mergeSchemas(
			sec.getFrameObject(input2.getName())));
	}

	private static class ReduceSideColumnsFunction implements Function<Tuple2<FrameBlock, FrameBlock>, FrameBlock> 
	{
		private static final long serialVersionUID = -97824903649667646L;

		private boolean _cbind = true;
				
		public ReduceSideColumnsFunction(boolean cbind) {
			_cbind = cbind;
		}
		
		@Override
		public FrameBlock call(Tuple2<FrameBlock, FrameBlock> arg0) {
			FrameBlock left = arg0._1();
			FrameBlock right = arg0._2();
			return left.append(right, new FrameBlock(), _cbind);
		}
	}

	private static class ReduceSideAppendRowsFunction implements PairFunction<Tuple2<Long, FrameBlock>, Long, FrameBlock> 
	{
		private static final long serialVersionUID = 1723795153048336791L;

		private long _offset;
				
		public ReduceSideAppendRowsFunction(long offset) {
			_offset = offset;
		}
		
		@Override
		public Tuple2<Long,FrameBlock> call(Tuple2<Long, FrameBlock> arg0)
			throws Exception 
		{
			return new Tuple2<>(arg0._1()+_offset, arg0._2());
		}
	}

	private static class ReduceSideAppendAlignFunction implements PairFunction<Tuple2<Long, FrameBlock>, Long, FrameBlock> 
	{
		private static final long serialVersionUID = 5850400295183766409L;

		private long _rows;
				
		public ReduceSideAppendAlignFunction(long rows) {
			_rows = rows;
		}
		
		@Override
		public Tuple2<Long,FrameBlock> call(Tuple2<Long, FrameBlock> arg0)
			throws Exception 
		{
			FrameBlock resultBlock = new FrameBlock(arg0._2().getSchema());
			long index = (arg0._1()/OptimizerUtils.DEFAULT_FRAME_BLOCKSIZE)*OptimizerUtils.DEFAULT_FRAME_BLOCKSIZE+1;
			int maxRows = (int) (_rows - index+1 >= OptimizerUtils.DEFAULT_FRAME_BLOCKSIZE?OptimizerUtils.DEFAULT_FRAME_BLOCKSIZE:_rows - index+1);
			resultBlock.ensureAllocatedColumns(maxRows);
			resultBlock = resultBlock.leftIndexingOperations(arg0._2(), 0, maxRows-1, 0, arg0._2().getNumColumns()-1, new FrameBlock());
			return new Tuple2<>(index, resultBlock);
		}
	}

}
