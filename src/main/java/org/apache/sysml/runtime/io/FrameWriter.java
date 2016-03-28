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

package org.apache.sysml.runtime.io;

import java.io.IOException;
import java.util.List;

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.matrix.data.FrameBlock;

/**
 * Base class for all format-specific frame writers. Every writer is required to implement the basic 
 * write functionality but might provide additional custom functionality. Any non-default parameters
 * (e.g., CSV read properties) should be passed into custom constructors. There is also a factory
 * for creating format-specific writers. 
 * 
 */
public abstract class FrameWriter 
{

	/**
	 * 
	 * @param src
	 * @param fname
	 * @param rlen
	 * @param clen
	 * @return
	 * @throws IOException
	 * @throws DMLUnsupportedOperationException 
	 * @throws DMLRuntimeException 
	 */
	public abstract void writeFrameToHDFS( FrameBlock src, String fname, long rlen, long clen )
		throws IOException, DMLRuntimeException, DMLUnsupportedOperationException;
	
	/**
	 * 
	 * @param schema
	 * @param names
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static FrameBlock[] createFrameBlocksForReuse( List<ValueType> schema, List<String> names, long rlen ) 
		throws DMLRuntimeException
	{
		FrameBlock frameBlock[] = new FrameBlock[1];
		frameBlock[0] = new FrameBlock(schema, names);

		frameBlock[0].ensureAllocatedColumns((int)rlen);
		return frameBlock;
	}
	
	/**
	 * 
	 * @param blocks
	 * @return
	 */
	public static FrameBlock getFrameBlockForReuse( FrameBlock[] blocks) //TODO do we need this function?
	{
		return blocks[ 0 ];
	}
}
