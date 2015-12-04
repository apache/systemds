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

import scala.Tuple2;

import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;

/**
 * Function class find the tuple that matches the given MatrixIndexes.
 */
public class FindMatrixBlockFromMatrixIndexes implements Function<Tuple2<MatrixIndexes,MatrixBlock>, Boolean> 
{
	private static final long serialVersionUID = 3579786335136549745L;
	MatrixIndexes _key = null;
	
	public FindMatrixBlockFromMatrixIndexes(MatrixIndexes k) { 
		_key = k;
	}

	@Override
	public Boolean call(Tuple2<MatrixIndexes, MatrixBlock> v1)
			throws Exception {
		return (v1._1().compareTo(_key) == 0);
	}
}
