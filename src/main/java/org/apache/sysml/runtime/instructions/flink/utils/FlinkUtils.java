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

package org.apache.sysml.runtime.instructions.flink.utils;

import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.util.UtilFunctions;

import java.util.ArrayList;

public class FlinkUtils {
	/**
	 * @param env
	 * @param mc
	 * @return
	 */
	public static DataSet<Tuple2<MatrixIndexes, MatrixBlock>> getEmptyBlockDataSet(ExecutionEnvironment env,
																				   MatrixCharacteristics mc) {
		//create all empty blocks
		ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> list = new ArrayList<Tuple2<MatrixIndexes, MatrixBlock>>();
		int nrblks = (int) Math.ceil((double) mc.getRows() / mc.getRowsPerBlock());
		int ncblks = (int) Math.ceil((double) mc.getCols() / mc.getColsPerBlock());
		for (long r = 1; r <= nrblks; r++)
			for (long c = 1; c <= ncblks; c++) {
				int lrlen = UtilFunctions.computeBlockSize(mc.getRows(), r, mc.getRowsPerBlock());
				int lclen = UtilFunctions.computeBlockSize(mc.getCols(), c, mc.getColsPerBlock());
				MatrixIndexes ix = new MatrixIndexes(r, c);
				MatrixBlock mb = new MatrixBlock(lrlen, lclen, true);
				list.add(new Tuple2<MatrixIndexes, MatrixBlock>(ix, mb));
			}

		//create dataset of in-memory list
		return env.fromCollection(list);
	}
}
