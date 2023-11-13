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

package org.apache.sysds.runtime.instructions.spark.utils;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import scala.Tuple2;
import scala.Tuple5;


public class FrameRDDAggregateUtils 
{
	public static Tuple2<Boolean, Integer> checkRowAlignment(JavaPairRDD<Long,FrameBlock> in, int blen){
		JavaRDD<Tuple5<Boolean, Long, Integer, Integer, Boolean>> row_rdd = in.map((Function<Tuple2<Long, FrameBlock>, Tuple5<Boolean, Long, Integer, Integer, Boolean>>) in1 -> {
			long key = in1._1();
			FrameBlock blk = in1._2();
			return new Tuple5<>(true, key, blen == -1 ? blk.getNumRows() : blen, blk.getNumRows(), true);
		});
		Tuple5<Boolean, Long, Integer, Integer, Boolean> result = row_rdd.fold(null, (Function2<Tuple5<Boolean, Long, Integer, Integer, Boolean>, Tuple5<Boolean, Long, Integer, Integer, Boolean>, Tuple5<Boolean, Long, Integer, Integer, Boolean>>) (in1, in2) -> {
			//easy evaluation
			if (in1 == null)
				return in2;
			if (in2 == null)
				return in1;
			if (!in1._1() || !in2._1())
				return new Tuple5<>(false, null, null, null, null);

			//default evaluation
			int in1_max = in1._3();
			int in1_min = in1._4();
			long in1_min_index = in1._2(); //Index of Block with min nr rows --> Block with largest index ( --> last block index)
			int in2_max = in2._3();
			int in2_min = in2._4();
			long in2_min_index = in2._2();

			boolean in1_isSingleBlock = in1._5();
			boolean in2_isSingleBlock = in2._5();
			boolean min_index_comp = in1_min_index > in2_min_index;

			if (in1_max == in2_max) {
				if (in1_min == in1_max) {
					if (in2_min == in2_max)
						return new Tuple5<>(true, min_index_comp ? in1_min_index : in2_min_index, in1_max, in1_max, false);
					else if (!min_index_comp)
						return new Tuple5<>(true, in2_min_index, in1_max, in2_min, false);
					//else: in1_min_index > in2_min_index -->  in2 is not aligned
				} else {
					if (in2_min == in2_max)
						if (min_index_comp)
							return new Tuple5<>(true, in1_min_index, in1_max, in1_min, false);
					//else: in1_min_index < in2_min_index -->  in1 is not aligned
					//else: both contain blocks with less blocks than max
				}
			} else {
				if (in1_max > in2_max && in1_min == in1_max && in2_isSingleBlock && in1_min_index < in2_min_index)
					return new Tuple5<>(true, in2_min_index, in1_max, in2_min, false);
				/* else:
				in1_min != in1_max -> both contain blocks with less blocks than max
				!in2_isSingleBlock -> in2 contains at least 2 blocks with less blocks than in1's max
				in1_min_index > in2_min_index -> in2's min block != lst block
				 */
				if (in1_max < in2_max && in2_min == in2_max && in1_isSingleBlock && in2_min_index < in1_min_index)
					return new Tuple5<>(true, in1_min_index, in2_max, in1_min, false);
			}
			return new Tuple5<>(false, null, null, null, null);
		});
		return new Tuple2<>(result._1(), result._3()) ;
	}

	public static JavaPairRDD<Long, FrameBlock> mergeByKey( JavaPairRDD<Long, FrameBlock> in )
	{
		//use combine by key to avoid unnecessary deep block copies, i.e.
		//create combiner block once and merge remaining blocks in-place.
 		return in.combineByKey( 
 				new CreateBlockCombinerFunction(), 
			    new MergeBlocksFunction(false), 
			    new MergeBlocksFunction(false) );
	}

	private static class CreateBlockCombinerFunction implements Function<FrameBlock, FrameBlock> 
	{
		private static final long serialVersionUID = -4445167244905540494L;

		@Override
		public FrameBlock call(FrameBlock arg0) 
			throws Exception 
		{
			//create deep copy of given block
			return new FrameBlock(arg0);
		}	
	}

	private static class MergeBlocksFunction implements Function2<FrameBlock, FrameBlock, FrameBlock> 
	{		
		private static final long serialVersionUID = 7807210434431147007L;
		
		private boolean _deep = false;
		
		public MergeBlocksFunction(boolean deep) {
			_deep = deep;
		}

		@Override
		public FrameBlock call(FrameBlock b1, FrameBlock b2) 
			throws Exception 
		{
			// sanity check input dimensions
			if (b1.getNumRows() != b2.getNumRows() || b1.getNumColumns() != b2.getNumColumns()) {
				throw new DMLRuntimeException("Mismatched frame block sizes for: "
						+ b1.getNumRows() + " " + b1.getNumColumns() + " "
						+ b2.getNumRows() + " " + b2.getNumColumns());
			}

			// execute merge (never pass by reference)
			FrameBlock ret = _deep ? new FrameBlock(b1) : b1;
			ret.merge(b2);
			return ret;
		}
	}	
}
