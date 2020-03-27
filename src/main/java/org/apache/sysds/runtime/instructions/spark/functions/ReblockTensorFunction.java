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

package org.apache.sysds.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.data.TensorIndexes;
import org.apache.sysds.runtime.meta.TensorCharacteristics;
import org.apache.sysds.runtime.util.UtilFunctions;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;


public class ReblockTensorFunction implements PairFlatMapFunction<Tuple2<TensorIndexes, TensorBlock>, TensorIndexes, TensorBlock> {
	private static final long serialVersionUID = 9118830682358813489L;
	
	private int _numDims;
	private long _newBlen;
	
	public ReblockTensorFunction(int numDims, long newBlen) {
		_numDims = numDims;
		_newBlen = newBlen;
	}
	
	@Override
	public Iterator<Tuple2<TensorIndexes, TensorBlock>> call(Tuple2<TensorIndexes, TensorBlock> arg0)
			throws Exception {
		TensorIndexes ti = arg0._1();
		TensorBlock tb = arg0._2();
		TensorCharacteristics tc = new TensorCharacteristics(tb.getLongDims(), (int) _newBlen);
		
		long[] tensorIndexes = new long[_numDims];
		for (int i = 0; i < tb.getNumDims(); i++) {
			tensorIndexes[i] = 1 + (ti.getIndex(i) - 1) * tc.getNumBlocks(i);
		}
		Arrays.fill(tensorIndexes, tb.getNumDims(), tensorIndexes.length, 1);
		long[] zeroBasedTensorIndexes = new long[tb.getNumDims()];
		Arrays.fill(zeroBasedTensorIndexes, 1);
		
		ArrayList<Tuple2<TensorIndexes, TensorBlock>> retVal = new ArrayList<>();
		long numBlocks = tc.getNumBlocks();
		int[] offsets = new int[tb.getNumDims()];
		for (int i = 0; i < numBlocks; i++) {
			int[] dims = new int[tb.getNumDims()];
			UtilFunctions.computeSliceInfo(tc, zeroBasedTensorIndexes, dims, offsets);
			TensorBlock outBlock;
			if (tb.isBasic())
				outBlock = new TensorBlock(tb.getValueType(), dims);
			else {
				ValueType[] schema = new ValueType[dims[1]];
				System.arraycopy(tb.getSchema(), offsets[1], schema, 0, dims[1]);
				outBlock = new TensorBlock(schema, dims);
			}
			tb.slice(offsets, outBlock);
			retVal.add(new Tuple2<>(new TensorIndexes(tensorIndexes), outBlock));
			UtilFunctions.computeNextTensorIndexes(tc, tensorIndexes);
			UtilFunctions.computeNextTensorIndexes(tc, zeroBasedTensorIndexes);
		}
		return retVal.iterator();
	}
}
