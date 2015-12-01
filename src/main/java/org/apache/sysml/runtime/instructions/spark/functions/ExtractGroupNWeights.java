/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.runtime.instructions.spark.functions;

import java.util.ArrayList;

import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;

import org.apache.sysml.runtime.matrix.data.IJV;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.SparseRowsIterator;
import org.apache.sysml.runtime.matrix.data.WeightedCell;
import org.apache.sysml.runtime.util.UtilFunctions;

public class ExtractGroupNWeights implements PairFlatMapFunction<Tuple2<MatrixIndexes,Tuple2<Iterable<Tuple2<MatrixBlock,MatrixBlock>>,Iterable<MatrixBlock>>>, Long, WeightedCell> {

	private static final long serialVersionUID = -188180042997588072L;

	@Override
	public Iterable<Tuple2<Long, WeightedCell>> call(
			Tuple2<MatrixIndexes, Tuple2<Iterable<Tuple2<MatrixBlock, MatrixBlock>>, Iterable<MatrixBlock>>> arg)
			throws Exception {
		MatrixBlock group = null;
		MatrixBlock target = null;
		for(Tuple2<MatrixBlock, MatrixBlock> kv : arg._2._1) {
			if(group == null) {
				group = kv._1;
				target = kv._2;
			}
			else {
				throw new Exception("More than 1 block with same MatrixIndexes");
			}
		}
		MatrixBlock weight = null;
		for(MatrixBlock blk : arg._2._2) {
			if(weight == null) {
				weight = blk;
			}
			else {
				throw new Exception("More than 1 block with same MatrixIndexes");
			}
		}
		
		ArrayList<Double> groupIDs = getColumn(group);
		ArrayList<Double> values = getColumn(target);
		ArrayList<Double> w = getColumn(weight);
		ArrayList<Tuple2<Long, WeightedCell>> groupValuePairs = new ArrayList<Tuple2<Long, WeightedCell>>();
		
		if(groupIDs != null) {
			if(groupIDs.size() != values.size() || groupIDs.size() != w.size()) {
				throw new Exception("The blocksize for group, weight and target block are mismatched: " 
						+ groupIDs.size()  + " != " + values.size() + " || " + groupIDs.size() + " != " + w.size());
			}
			for(int i = 0; i < groupIDs.size(); i++) {
				WeightedCell weightedCell = new WeightedCell();
				try {
					weightedCell.setValue(values.get(i));
				}
				catch(Exception e) {
					weightedCell.setValue(0);
				}
				try {
					weightedCell.setWeight(w.get(i));
				}
				catch(Exception e) {
					weightedCell.setValue(1);
				}
				long groupVal = UtilFunctions.toLong(groupIDs.get(i));
				if(groupVal < 1) {
					throw new Exception("Expected group values to be greater than equal to 1 but found " + groupVal);
				}
				groupValuePairs.add(new Tuple2<Long, WeightedCell>(groupVal, weightedCell));
			}
		}
		else {
			throw new Exception("group ids block shouldn't be empty");
		}
		
		return groupValuePairs;
	}
	
	public ArrayList<Double> getColumn(MatrixBlock blk) throws Exception {
		ArrayList<Double> retVal = new ArrayList<Double>();
		if(blk != null) {
			if (blk.isInSparseFormat()) {
				SparseRowsIterator iter = blk.getSparseRowsIterator();
				while( iter.hasNext() ) {
					IJV cell = iter.next();
					retVal.add(cell.v);
				}
			}
			else {
				double[] valuesInBlock = blk.getDenseArray();
				for(int i = 0; i < valuesInBlock.length; i++) {
					retVal.add(valuesInBlock[i]);
				}
			}
		}
		return retVal;
	}
	
}
