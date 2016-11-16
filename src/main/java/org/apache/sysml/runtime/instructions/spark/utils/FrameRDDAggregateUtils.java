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

package org.apache.sysml.runtime.instructions.spark.utils;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.FrameBlock;


public class FrameRDDAggregateUtils 
{

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
