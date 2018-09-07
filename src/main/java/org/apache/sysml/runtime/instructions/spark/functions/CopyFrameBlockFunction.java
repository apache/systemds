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

import org.apache.spark.api.java.function.Function;
import org.apache.sysml.runtime.matrix.data.FrameBlock;

/**
 * General purpose copy function for binary block rdds. This function can be used in
 * mapValues (copy frame blocks). It supports both deep and shallow copies of values.
 * 
 */
public class CopyFrameBlockFunction implements Function<FrameBlock,FrameBlock> 
{
	private static final long serialVersionUID = 612972882700587381L;
	
	private boolean _deepCopy = true;
	
	public CopyFrameBlockFunction() {
		this(true);
	}
	
	public CopyFrameBlockFunction(boolean deepCopy) {
		_deepCopy = deepCopy;
	}

	@Override
	public FrameBlock call(FrameBlock arg0)
		throws Exception 
	{
		if( _deepCopy )
			return new FrameBlock(arg0);
		else
			return arg0;
	}
}