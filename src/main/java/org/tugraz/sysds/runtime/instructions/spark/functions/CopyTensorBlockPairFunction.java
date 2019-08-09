/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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
package org.tugraz.sysds.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.tugraz.sysds.runtime.data.HomogTensor;
import org.tugraz.sysds.runtime.data.TensorIndexes;
import org.tugraz.sysds.runtime.instructions.spark.data.LazyIterableIterator;
import scala.Tuple2;

import java.util.Iterator;

/**
 * General purpose copy function for binary block rdds. This function can be used in
 * mapToPair (copy tensor indexes and blocks). It supports both deep and shallow copies
 * of key/value pairs.
 */
public class CopyTensorBlockPairFunction implements PairFlatMapFunction<Iterator<Tuple2<TensorIndexes, HomogTensor>>, TensorIndexes, HomogTensor> {

	private static final long serialVersionUID = 605514365345997070L;
	private boolean _deepCopy;

	public CopyTensorBlockPairFunction() {
		this(true);
	}

	public CopyTensorBlockPairFunction(boolean deepCopy) {
		_deepCopy = deepCopy;
	}

	@Override
	public LazyIterableIterator<Tuple2<TensorIndexes, HomogTensor>> call(Iterator<Tuple2<TensorIndexes, HomogTensor>> arg0)
			throws Exception {
		return new CopyBlockPairIterator(arg0);
	}

	private class CopyBlockPairIterator extends LazyIterableIterator<Tuple2<TensorIndexes, HomogTensor>> {
		public CopyBlockPairIterator(Iterator<Tuple2<TensorIndexes, HomogTensor>> iter) {
			super(iter);
		}

		@Override
		protected Tuple2<TensorIndexes, HomogTensor> computeNext(Tuple2<TensorIndexes, HomogTensor> arg) {
			if (_deepCopy) {
				TensorIndexes ix = new TensorIndexes(arg._1());
				HomogTensor block;
				// TODO: always create deep copies in more memory-efficient CSR representation
				//  if block is already in sparse format
				block = new HomogTensor(arg._2());
				return new Tuple2<>(ix, block);
			} else {
				return arg;
			}
		}
	}
}