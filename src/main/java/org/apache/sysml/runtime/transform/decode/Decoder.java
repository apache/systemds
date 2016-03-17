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

package org.apache.sysml.runtime.transform.decode;

import java.util.List;

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

/**
 * Base class for all transform decoders providing both a row and block
 * interface for decoding matrices to frames.
 * 
 */
public abstract class Decoder 
{	
	protected List<ValueType> _schema = null;
	
	protected Decoder( List<ValueType> schema ) {
		_schema = schema;
	}
	
	/**
	 * Row decode API converting a matrix row into a frame row
	 * of the specified decoder schema.
	 * 
	 * @param in
	 * @param out
	 */
	public abstract void decode(double[] in, Object[] out);
	
	/**
	 * Block decode API converting a matrix block into a frame block.
	 * 
	 * @param in
	 * @param out
	 * 
	 * @return returns given output frame block for convenience
	 */
	public abstract FrameBlock decode(MatrixBlock in, FrameBlock out);
}
