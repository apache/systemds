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

package org.apache.sysml.runtime.instructions.flink.functions;

import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;

public class FilterNonEmptyBlocksFunction implements FilterFunction<Tuple2<MatrixIndexes, MatrixBlock>> {

	@Override
	public boolean filter(Tuple2<MatrixIndexes, MatrixBlock> arg0)
			throws Exception {
		//always keep 1-1 block in order to prevent empty datasets
		boolean ix1 = (arg0.f0.getRowIndex() == 1
				&& arg0.f0.getColumnIndex() == 1);

		//returns true for non-empty matrix blocks
		return !arg0.f1.isEmptyBlock(false) || ix1;
	}


}
