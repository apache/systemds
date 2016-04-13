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

import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.common.operators.Order;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Collector;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class IndexUtils {

	private static <T> DataSet<Tuple2<Integer, Long>> countElements(DataSet<Tuple2<Integer, T>> input) {
		return input.mapPartition(new RichMapPartitionFunction<Tuple2<Integer, T>, Tuple2<Integer, Long>>() {
			@Override
			public void mapPartition(Iterable<Tuple2<Integer, T>> iterable,
									 Collector<Tuple2<Integer, Long>> collector) throws Exception {
				Iterator<Tuple2<Integer, T>> itr = iterable.iterator();
				if (itr.hasNext()) {
					int splitIndex = itr.next().f0;
					long counter = 1L;
					for (; itr.hasNext(); ++counter) {
						Tuple2<Integer, T> value = itr.next();
						if (value.f0 != splitIndex) {
							collector.collect(new Tuple2<Integer, Long>(splitIndex, counter));
							splitIndex = value.f0;
							counter = 0L;
						}
					}
					collector.collect(new Tuple2<Integer, Long>(splitIndex, counter));
				}
			}
		});
	}

	public static <T> DataSet<Tuple2<Long, T>> zipWithRowIndex(DataSet<Tuple2<Integer, T>> input) {
		DataSet<Tuple2<Integer, Long>> elementCount = countElements(input).sortPartition(0,
				Order.ASCENDING).setParallelism(1);

		return input.mapPartition(new RichMapPartitionFunction<Tuple2<Integer, T>, Tuple2<Long, T>>() {
			private long[] splitOffsets;
			private long[] splitCounts;
			private final Map<Integer, Integer> idTable = new HashMap<Integer, Integer>();

			@Override
			public void open(Configuration parameters) throws Exception {
				final List<Tuple2<Integer, Long>> tmp = this.getRuntimeContext().<Tuple2<Integer, Long>>getBroadcastVariable(
						"counts");

				// remap the splitIDs to start from 0 sequentially (necessary for local splits where dop > numSplits)
				synchronized (idTable) {
					for (int i = 0; i < tmp.size(); i++) {
						idTable.put(tmp.get(i).f0, i);
					}
				}

				this.splitOffsets = new long[tmp.size()];
				this.splitCounts = new long[tmp.size()];
				for (int i = 1; i < tmp.size(); i++) {
					splitOffsets[i] = tmp.get(i - 1).f1 + splitOffsets[i - 1];
				}
			}

			@Override
			public void mapPartition(Iterable<Tuple2<Integer, T>> values,
									 Collector<Tuple2<Long, T>> out) throws Exception {
				for (Tuple2<Integer, T> value : values) {
					Tuple2<Long, T> t = new Tuple2<Long, T>(
							this.splitOffsets[idTable.get(value.f0)] + this.splitCounts[idTable.get(value.f0)]++,
							value.f1);
					out.collect(t);
				}
			}
		}).withBroadcastSet(elementCount, "counts");

	}
}
