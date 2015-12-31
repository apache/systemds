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

import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.WeightedCell;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.util.UtilFunctions;

public class ExtractGroup  implements PairFlatMapFunction<Tuple2<MatrixIndexes,Tuple2<MatrixBlock, MatrixBlock>>, MatrixIndexes, WeightedCell> {

	private static final long serialVersionUID = -7059358143841229966L;

	private long _bclen = -1; 
	private long _ngroups = -1; 
	private Operator _op = null;
	
	public ExtractGroup( long bclen, long ngroups, Operator op ) {
		_bclen = bclen;
		_ngroups = ngroups;
		_op = op;
	}
	
	@Override
	public Iterable<Tuple2<MatrixIndexes, WeightedCell>> call(
			Tuple2<MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>> arg)
			throws Exception 
	{
		MatrixIndexes ix = arg._1;
		MatrixBlock group = arg._2._1;
		MatrixBlock target = arg._2._2;
		
		//sanity check matching block dimensions
		if(group.getNumRows() != target.getNumRows()) {
			throw new Exception("The blocksize for group and target blocks are mismatched: " + group.getNumRows()  + " != " + target.getNumRows());
		}
		
		//output weighted cells
		ArrayList<Tuple2<MatrixIndexes, WeightedCell>> groupValuePairs = new ArrayList<Tuple2<MatrixIndexes, WeightedCell>>();
		long coloff = (ix.getColumnIndex()-1)*_bclen;
		
		//local pre-aggregation for sum w/ known output dimensions
		if(_op instanceof AggregateOperator && _ngroups > 0 
			&& OptimizerUtils.isValidCPDimensions(_ngroups, target.getNumColumns()) ) 
		{
			MatrixBlock tmp = group.groupedAggOperations(target, null, new MatrixBlock(), (int)_ngroups, _op);
			
			for(int i=0; i<tmp.getNumRows(); i++) {
				for( int j=0; j<tmp.getNumColumns(); j++ ) {
					double tmpval = tmp.quickGetValue(i, j);
					if( tmpval != 0 ) {
						WeightedCell weightedCell = new WeightedCell();
						weightedCell.setValue(tmpval);
						weightedCell.setWeight(1);
						MatrixIndexes ixout = new MatrixIndexes(i+1,coloff+j+1);
						groupValuePairs.add(new Tuple2<MatrixIndexes, WeightedCell>(ixout, weightedCell));
					}
				}
			}
		}
		//general case without pre-aggregation
		else 
		{
			for(int i = 0; i < group.getNumRows(); i++) {
				long groupVal = UtilFunctions.toLong(group.quickGetValue(i, 0));
				if(groupVal < 1) {
					throw new Exception("Expected group values to be greater than equal to 1 but found " + groupVal);
				}
				for( int j=0; j<target.getNumColumns(); j++ ) {
					WeightedCell weightedCell = new WeightedCell();
					weightedCell.setValue(target.quickGetValue(i, j));
					weightedCell.setWeight(1);
					MatrixIndexes ixout = new MatrixIndexes(groupVal,coloff+j+1);
					groupValuePairs.add(new Tuple2<MatrixIndexes, WeightedCell>(ixout, weightedCell));
				}
			}
		}
		
		return groupValuePairs;
	}
}
