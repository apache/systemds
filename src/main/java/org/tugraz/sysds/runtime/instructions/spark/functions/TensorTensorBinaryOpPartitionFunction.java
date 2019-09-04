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
import org.tugraz.sysds.runtime.instructions.spark.data.LazyIterableIterator;
import org.tugraz.sysds.runtime.instructions.spark.data.PartitionedBroadcast;
import org.tugraz.sysds.runtime.matrix.operators.BinaryOperator;
import scala.Tuple2;

import java.util.Iterator;

public class TensorTensorBinaryOpPartitionFunction implements PairFlatMapFunction<Iterator<Tuple2<TensorIndexes, TensorBlock>>, TensorIndexes, TensorBlock> {

	private static final long serialVersionUID = 8029096658247920867L;
	private BinaryOperator _op;
	private PartitionedBroadcast<TensorBlock> _ptV;
	private boolean[] _replicateDim;

	public TensorTensorBinaryOpPartitionFunction(BinaryOperator op, PartitionedBroadcast<TensorBlock> binput,
			boolean[] replicateDim) {
		_op = op;
		_ptV = binput;
		_replicateDim = replicateDim;
	}

	@Override
	public LazyIterableIterator<Tuple2<TensorIndexes, TensorBlock>> call(
			Iterator<Tuple2<TensorIndexes, TensorBlock>> arg0)
			throws Exception {
		return new MapBinaryPartitionIterator(arg0);
	}

	/**
	 * Lazy mbinary iterator to prevent materialization of entire partition output in-memory.
	 * The implementation via mapPartitions is required to preserve partitioning information,
	 * which is important for performance.
	 */
	private class MapBinaryPartitionIterator extends LazyIterableIterator<Tuple2<TensorIndexes, TensorBlock>> {
		public MapBinaryPartitionIterator(Iterator<Tuple2<TensorIndexes, TensorBlock>> in) {
			super(in);
		}

		@Override
		protected Tuple2<TensorIndexes, TensorBlock> computeNext(Tuple2<TensorIndexes, TensorBlock> arg) {
			//unpack partition key-value pairs
			TensorIndexes ix = arg._1();
			TensorBlock in1 = arg._2();

			//get the rhs block
			int[] index = new int[in1.getNumDims()];
			for (int i = 0; i < index.length; i++) {
				if (_replicateDim[i])
					index[i] = 1;
				else
					index[i] = (int) ix.getIndex(i);
			}
			TensorBlock in2 = _ptV.getBlock(index);

			//execute the binary operation
			TensorBlock ret = in1.binaryOperations(_op, in2, new TensorBlock());
			return new Tuple2<>(ix, ret);
		}
	}
}
