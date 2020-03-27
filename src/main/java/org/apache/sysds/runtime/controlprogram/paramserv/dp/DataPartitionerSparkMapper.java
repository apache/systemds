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

package org.apache.sysds.runtime.controlprogram.paramserv.dp;

import java.io.Serializable;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import scala.Tuple2;

public class DataPartitionerSparkMapper implements PairFlatMapFunction<Tuple2<Long, Tuple2<MatrixBlock, MatrixBlock>>, Integer, Tuple2<Long, Tuple2<MatrixBlock, MatrixBlock>>>, Serializable {

	private static final long serialVersionUID = 1710721606050403296L;
	private int _workersNum;

	private SparkDataPartitioner _dp;

	protected DataPartitionerSparkMapper() {
		// No-args constructor used for deserialization
	}

	public DataPartitionerSparkMapper(Statement.PSScheme scheme, int workersNum, SparkExecutionContext sec, int numEntries) {
		_workersNum = workersNum;
		_dp = new SparkDataPartitioner(scheme, sec, numEntries, workersNum);
	}

	/**
	 * Do data partitioning
	 * @param input RowBlockID {@literal =>} (features, labels)
	 * @return WorkerID {@literal =>} (rowBlockID, (single row features, single row labels))
	 * @throws Exception Some exception
	 */
	@Override
	public Iterator<Tuple2<Integer, Tuple2<Long, Tuple2<MatrixBlock, MatrixBlock>>>> call(Tuple2<Long,Tuple2<MatrixBlock,MatrixBlock>> input)
			throws Exception {
		List<Tuple2<Integer, Tuple2<Long,Tuple2<MatrixBlock,MatrixBlock>>>> partitions = new LinkedList<>();
		MatrixBlock features = input._2._1;
		MatrixBlock labels = input._2._2;
		DataPartitionSparkScheme.Result result = _dp.doPartitioning(_workersNum, features, labels, input._1);
		for (int i = 0; i < result.pFeatures.size(); i++) {
			Tuple2<Integer, Tuple2<Long, MatrixBlock>> ft = result.pFeatures.get(i);
			Tuple2<Integer, Tuple2<Long, MatrixBlock>> lt = result.pLabels.get(i);
			partitions.add(new Tuple2<>(ft._1, new Tuple2<>(ft._2._1, new Tuple2<>(ft._2._2, lt._2._2))));
		}
		return partitions.iterator();
	}
}
