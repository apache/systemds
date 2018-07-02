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

package org.apache.sysml.runtime.controlprogram.paramserv.spark;

import java.io.Serializable;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

import scala.Tuple2;

public class DataPartitionerSparkMapper implements PairFlatMapFunction<Tuple2<Long,Tuple2<MatrixBlock,MatrixBlock>>, Integer, Tuple2<MatrixBlock, MatrixBlock>>, Serializable {

	private static final long serialVersionUID = 1710721606050403296L;
	private int _workersNum;

	private DataPartitionSparkScheme _scheme;

	@SuppressWarnings("unused")
	protected DataPartitionerSparkMapper() {
		// No-args constructor used for deserialization
	}

	public DataPartitionerSparkMapper(Statement.PSScheme scheme, int workersNum) {
		_workersNum = workersNum;
		switch (scheme) {
			case DISJOINT_CONTIGUOUS:
				_scheme = new DCSparkScheme();
				break;
			case DISJOINT_ROUND_ROBIN:
				_scheme = new DRRSparkScheme();
				break;
			case DISJOINT_RANDOM:
				_scheme = new DRSparkScheme();
				break;
			case OVERLAP_RESHUFFLE:
				_scheme = new ORSparkScheme();
				break;
		}

	}

	/**
	 *
	 * @param input Row Block ID -> "features, labels"
	 * @return WorkerID -> partitioned "features, labels"
	 * @throws Exception Some exception
	 */
	@Override
	public Iterator<Tuple2<Integer, Tuple2<MatrixBlock, MatrixBlock>>> call(Tuple2<Long,Tuple2<MatrixBlock,MatrixBlock>> input)
			throws Exception {
		List<Tuple2<Integer, Tuple2<MatrixBlock, MatrixBlock>>> partitions = new LinkedList<>();
		MatrixBlock features = input._2._1;
		MatrixBlock labels = input._2._2;
		DataPartitionSparkScheme.Result result = _scheme.doPartitioning(_workersNum, features, labels);
		for (int id = 0; id < _workersNum; id++) {
			partitions.add(new Tuple2<>(id, new Tuple2<>(result.pFeatures.get(id), result.pLabels.get(id))));
		}
		return partitions.iterator();
	}
}
