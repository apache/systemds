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
package org.apache.sysml.runtime.instructions.spark.functions;

import org.apache.hadoop.io.LongWritable;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import org.apache.sysml.runtime.matrix.data.FrameBlock;

/**
 * General purpose copy function for binary block rdds. This function can be used in
 * mapToPair (copy frame indexes and blocks). It supports both deep and shallow copies 
 * of key/value pairs.
 * 
 */
public class CopyFrameBlockPairFunction implements PairFunction<Tuple2<LongWritable, FrameBlock>,Long, FrameBlock>
{
	private static final long serialVersionUID = -4686652006382558021L;
	
	private boolean _deepCopy = true;
	
	public CopyFrameBlockPairFunction() {
		this(true);
	}
	
	public CopyFrameBlockPairFunction(boolean deepCopy) {
		_deepCopy = deepCopy;
	}

	@Override
	public Tuple2<Long, FrameBlock> call(Tuple2<LongWritable, FrameBlock> arg0) 
		throws Exception 
	{	
		if( _deepCopy ) {
			FrameBlock block = new FrameBlock(arg0._2());
			return new Tuple2<Long,FrameBlock>(arg0._1().get(), block);
		}
		else {
			return new Tuple2<Long,FrameBlock>(arg0._1().get(), arg0._2());
		}
	}
}