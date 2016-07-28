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

package org.apache.sysml.runtime.controlprogram.caching;

import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

/**
 * Factory to create instances of matrix/frame blocks given
 * internal codes.
 * 
 */
public class CacheBlockFactory 
{
	/**
	 * 
	 * @param code
	 * @return
	 */
	public static CacheBlock newInstance(int code) {
		switch( code ) {
			case 0: return new MatrixBlock();
			case 1: return new FrameBlock();
		}
		throw new RuntimeException("Unsupported cache block type: "+code);
	}
	
	/**
	 * 
	 * @param block
	 * @return
	 */
	public static int getCode(CacheBlock block) {
		if( block instanceof MatrixBlock )
			return 0;
		else if( block instanceof FrameBlock )
			return 1;
		throw new RuntimeException("Unsupported cache block type: "+block.getClass().getName());
	}
}
