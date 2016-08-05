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

import java.util.ArrayList;
import java.util.Iterator;

import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;

import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.WeightedCell;
import org.apache.sysml.runtime.util.UtilFunctions;

public class ExtractGroupNWeights implements PairFlatMapFunction<Tuple2<MatrixIndexes,Tuple2<Tuple2<MatrixBlock,MatrixBlock>,MatrixBlock>>, MatrixIndexes, WeightedCell> {

	private static final long serialVersionUID = -188180042997588072L;

	@Override
	public Iterator<Tuple2<MatrixIndexes, WeightedCell>> call(
			Tuple2<MatrixIndexes, Tuple2<Tuple2<MatrixBlock, MatrixBlock>, MatrixBlock>> arg)
		throws Exception 
	{
		MatrixBlock group = arg._2._1._1;
		MatrixBlock target = arg._2._1._2;
		MatrixBlock weight = arg._2._2;
		
		//sanity check matching block dimensions
		if(group.getNumRows() != target.getNumRows() || group.getNumRows()!=target.getNumRows()) {
			throw new Exception("The blocksize for group/target/weight blocks are mismatched: " + group.getNumRows()  + ", " + target.getNumRows() + ", " + weight.getNumRows());
		}
		
		//output weighted cells		
		ArrayList<Tuple2<MatrixIndexes, WeightedCell>> groupValuePairs = new ArrayList<Tuple2<MatrixIndexes, WeightedCell>>();
		for(int i = 0; i < group.getNumRows(); i++) {
			WeightedCell weightedCell = new WeightedCell();
			weightedCell.setValue(target.quickGetValue(i, 0));
			weightedCell.setWeight(weight.quickGetValue(i, 0));
			long groupVal = UtilFunctions.toLong(group.quickGetValue(i, 0));
			if(groupVal < 1) {
				throw new Exception("Expected group values to be greater than equal to 1 but found " + groupVal);
			}
			MatrixIndexes ix = new MatrixIndexes(groupVal, 1);
			groupValuePairs.add(new Tuple2<MatrixIndexes, WeightedCell>(ix, weightedCell));
		}
		
		return groupValuePairs.iterator();
	}
}
