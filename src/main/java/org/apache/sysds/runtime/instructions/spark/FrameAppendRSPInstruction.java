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
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.utils.FrameRDDAggregateUtils;
import org.apache.sysds.runtime.matrix.operators.Operator;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

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
		JavaPairRDD<Long,FrameBlock> out;
		long leftRows = sec.getDataCharacteristics(input1.getName()).getRows();

		out = appendFrameRSP(in1, in2, leftRows, _cbind);

		//put output RDD handle into symbol table
		updateBinaryAppendOutputDataCharacteristics(sec, _cbind);
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		sec.addLineageRDD(output.getName(), input2.getName());
		
		if(_cbind)
			//update schema of output with merged input schemas
			sec.getFrameObject(output.getName()).setSchema(
				sec.getFrameObject(input1.getName()).mergeSchemas(
				sec.getFrameObject(input2.getName())));
		else
			sec.getFrameObject(output.getName()).setSchema(sec.getFrameObject(input1.getName()).getSchema());
	}

	public static JavaPairRDD<Long, FrameBlock> appendFrameRSP(JavaPairRDD<Long, FrameBlock> in1, JavaPairRDD<Long, FrameBlock> in2, long leftRows, boolean cbind) {
		if(cbind) {
			//get in1 keys
			long[] row_indices = in1.keys().collect().stream().mapToLong(Long::longValue).toArray();
			Arrays.sort(row_indices);
			//Align the blocks of in2 on the blocks of in1
			JavaPairRDD<Long,FrameBlock> in2Aligned = in2.flatMapToPair(new ReduceSideAppendAlignToLHSFunction(row_indices, leftRows));
			in2Aligned = FrameRDDAggregateUtils.mergeByKey(in2Aligned);
			return in1.join(in2Aligned).mapValues(new ReduceSideColumnsFunction(cbind));
		} else {	//rbind
			JavaPairRDD<Long,FrameBlock> right = in2.mapToPair( new ReduceSideAppendRowsFunction(leftRows));
			return in1.union(right);
		}
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
			return left.append(right, _cbind);
		}
	}

	private static class ReduceSideAppendAlignToLHSFunction implements PairFlatMapFunction<Tuple2<Long, FrameBlock>, Long, FrameBlock>
	{
		private static final long serialVersionUID = 5850400295183766409L;

		private final long[] _indices;
		private final long lastIndex; //max_rows + 1

		public ReduceSideAppendAlignToLHSFunction(long[] indices, long max_rows) {
			_indices = indices;
			lastIndex = max_rows + 1;
		}

		@Override
		public Iterator<Tuple2<Long, FrameBlock>> call(Tuple2<Long, FrameBlock> arg0)
		{
			List<Tuple2<Long, FrameBlock>> aligned_blocks = new ArrayList<>();
			long indexRHS = arg0._1();
			FrameBlock fb = arg0._2();

			//find the block index ix in the LHS with the smallest index s.t. following LHS indix ix' > indexRHS >= ix
			//doing binary search
			int L = 0;
			int R = _indices.length - 1;
			int m;
			while(L <= R){
				m = (L + R) / 2;
				if(_indices[m] == indexRHS){
					R = m;
					break;
				}
				if(_indices[m] < indexRHS)
					L = m + 1;
        		else
					R = m - 1;
			}
			// search terminates if we have found the exact indexRHS or binary search reached the leaf nodes where
			// L == R (bucket size = 1) and m == L which implies that _indices[m+1] > indexRHS (otherwise we would
			// have considered this index in the search
			// if _indices[m] < indexRHS than m contains the index which fits our definition and R == m
			// else (m - 1) fits our definition which is stored and R = m - 1
			// therefore in all cases the correct position of the indexLHS  is stored in R
			long indexLHS = _indices[R];

			//assumes total num rows LHS == RHS
			long nextIndexLHS = R < _indices.length - 1? _indices[R+1] : this.lastIndex;
			int blkSizeLHS = (int) (nextIndexLHS -  indexLHS);
			int offsetLHS = (int) (indexRHS - indexLHS);
			int offsetRHS = 0;
			int sizeOfSlice = blkSizeLHS - offsetLHS;

			FrameBlock resultBlock = new FrameBlock(fb.getSchema());
			resultBlock.ensureAllocatedColumns(blkSizeLHS);

			int sizeOfRHS = fb.getNumRows();
			while(sizeOfSlice < sizeOfRHS){
				FrameBlock fb_sliced = fb.slice(offsetRHS, offsetRHS + sizeOfSlice - 1);
				resultBlock = resultBlock.leftIndexingOperations(fb_sliced,offsetLHS, offsetLHS + sizeOfSlice - 1, 0, fb.getNumColumns()-1, new FrameBlock());
				aligned_blocks.add(new Tuple2<>(indexLHS, resultBlock));
				resultBlock = new FrameBlock(fb.getSchema());
				if(R >= _indices.length - 1)
					throw new RuntimeException("Alignment Error while CBIND: LHS has fewer rows than RHS");
				indexLHS = nextIndexLHS;
				offsetRHS += sizeOfSlice;
				offsetLHS = 0;
				sizeOfRHS -= sizeOfSlice;
				R++;
				nextIndexLHS =  R < _indices.length - 1? _indices[R+1] : this.lastIndex;
				sizeOfSlice = (int) (nextIndexLHS -  indexLHS); //sizeOfSlice = blkSizeLHS
				resultBlock.ensureAllocatedColumns(sizeOfSlice);
			}
			//RHS fits into aligned LHS block
			if(offsetRHS != 0)
				fb = fb.slice(offsetRHS, offsetRHS + sizeOfRHS - 1);
			resultBlock = resultBlock.leftIndexingOperations(fb, offsetLHS, offsetLHS + fb.getNumRows() - 1, 0, fb.getNumColumns()-1, new FrameBlock());
			aligned_blocks.add(new Tuple2<>(indexLHS, resultBlock));

			return aligned_blocks.iterator();
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
