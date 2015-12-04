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

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;

public class UnflattenIterablesAfterCogroup implements PairFunction<Tuple2<MatrixIndexes,Tuple2<Iterable<MatrixBlock>,Iterable<MatrixBlock>>>, MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>> {

	private static final long serialVersionUID = 5367350062892272775L;

	@Override
	public Tuple2<MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>> call(
			Tuple2<MatrixIndexes, Tuple2<Iterable<MatrixBlock>, Iterable<MatrixBlock>>> arg)
			throws Exception {
		MatrixBlock left = null;
		MatrixBlock right = null;
		for(MatrixBlock blk : arg._2._1) {
			if(left == null) {
				left = blk;
			}
			else {
				throw new Exception("More than 1 block with same MatrixIndexes");
			}
		}
		for(MatrixBlock blk : arg._2._2) {
			if(right == null) {
				right = blk;
			}
			else {
				throw new Exception("More than 1 block with same MatrixIndexes");
			}
		}
		return new Tuple2<MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>>(arg._1, new Tuple2<MatrixBlock, MatrixBlock>(left, right));
	}
	
}