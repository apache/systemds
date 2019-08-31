/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.tugraz.sysds.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.tugraz.sysds.runtime.data.TensorBlock;
import org.tugraz.sysds.runtime.data.TensorIndexes;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;


public class ReplicateTensorFunction implements PairFlatMapFunction<Tuple2<TensorIndexes, TensorBlock>, TensorIndexes, TensorBlock> {
	private static final long serialVersionUID = 7181347334827684965L;

	private int _byDim;
	private long _numReplicas;

	public ReplicateTensorFunction(int byDim, long numReplicas) {
		_byDim = byDim;
		_numReplicas = numReplicas;
	}

	@Override
	public Iterator<Tuple2<TensorIndexes, TensorBlock>> call(Tuple2<TensorIndexes, TensorBlock> arg0)
			throws Exception {
		TensorIndexes ix = arg0._1();
		TensorBlock tb = arg0._2();

		//sanity check inputs
		if (ix.getIndex(_byDim) != 1 || tb.getDim(_byDim) > 1) {
			throw new Exception("Expected dimension " + _byDim + " to be 1 in ReplicateTensor");
		}

		ArrayList<Tuple2<TensorIndexes, TensorBlock>> retVal = new ArrayList<>();
		long[] indexes = ix.getIndexes();
		for (int i = 1; i <= _numReplicas; i++) {
			indexes[_byDim] = i;
			retVal.add(new Tuple2<>(new TensorIndexes(indexes), tb));
		}
		return retVal.iterator();
	}
}
