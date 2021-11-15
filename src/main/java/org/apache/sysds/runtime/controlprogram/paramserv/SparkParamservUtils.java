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

package org.apache.sysds.runtime.controlprogram.paramserv;

import java.util.LinkedList;

import org.apache.spark.Partitioner;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.paramserv.dp.DataPartitionerSparkAggregator;
import org.apache.sysds.runtime.controlprogram.paramserv.dp.DataPartitionerSparkMapper;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.utils.Statistics;

import scala.Tuple2;

public class SparkParamservUtils {
	
	/**
	 * Assemble the matrix of features and labels according to the rowID
	 *
	 * @param featuresRDD indexed features matrix block
	 * @param labelsRDD indexed labels matrix block
	 * @return Assembled rdd with rowID as key while matrix of features and labels as value (rowID {@literal ->} features, labels)
	 */
	public static JavaPairRDD<Long, Tuple2<MatrixBlock, MatrixBlock>> assembleTrainingData(JavaPairRDD<MatrixIndexes, MatrixBlock> featuresRDD, JavaPairRDD<MatrixIndexes, MatrixBlock> labelsRDD) {
		JavaPairRDD<Long, MatrixBlock> fRDD = groupMatrix(featuresRDD);
		JavaPairRDD<Long, MatrixBlock> lRDD = groupMatrix(labelsRDD);
		//TODO Add an additional physical operator which broadcasts the labels directly (broadcast join with features) if certain memory budgets are satisfied
		return fRDD.join(lRDD);
	}

	private static JavaPairRDD<Long, MatrixBlock> groupMatrix(JavaPairRDD<MatrixIndexes, MatrixBlock> rdd) {
		//TODO could use join and aggregation to avoid unnecessary shuffle introduced by reduceByKey
		return rdd.mapToPair(input -> new Tuple2<>(input._1.getRowIndex(), new Tuple2<>(input._1.getColumnIndex(), input._2)))
			.aggregateByKey(new LinkedList<Tuple2<Long, MatrixBlock>>(),
				(list, input) -> {
					list.add(input);
					return list;
				}, 
				(l1, l2) -> {
					l1.addAll(l2);
					l1.sort((o1, o2) -> o1._1.compareTo(o2._1));
					return l1;
				})
			.mapToPair(input -> {
				LinkedList<Tuple2<Long, MatrixBlock>> list = input._2;
				MatrixBlock result = list.get(0)._2;
				for (int i = 1; i < list.size(); i++) {
					result = ParamservUtils.cbindMatrix(result, list.get(i)._2);
				}
				return new Tuple2<>(input._1, result);
			});
	}

	@SuppressWarnings("unchecked")
	public static JavaPairRDD<Integer, Tuple2<MatrixBlock, MatrixBlock>> doPartitionOnSpark(SparkExecutionContext sec, MatrixObject features, MatrixObject labels, Statement.PSScheme scheme, int workerNum) {
		Timing tSetup = DMLScript.STATISTICS ? new Timing(true) : null;
		// Get input RDD
		JavaPairRDD<MatrixIndexes, MatrixBlock> featuresRDD = (JavaPairRDD<MatrixIndexes, MatrixBlock>)
			sec.getRDDHandleForMatrixObject(features, FileFormat.BINARY);
		JavaPairRDD<MatrixIndexes, MatrixBlock> labelsRDD = (JavaPairRDD<MatrixIndexes, MatrixBlock>)
			sec.getRDDHandleForMatrixObject(labels, FileFormat.BINARY);

		DataPartitionerSparkMapper mapper = new DataPartitionerSparkMapper(scheme, workerNum, sec, (int) features.getNumRows());
		JavaPairRDD<Integer, Tuple2<MatrixBlock, MatrixBlock>> result = 
			assembleTrainingData(featuresRDD, labelsRDD) // Combine features and labels into a pair (rowBlockID => (features, labels))
			.flatMapToPair(mapper) // Do the data partitioning on spark (workerID => (rowBlockID, (single row features, single row labels))
			// Aggregate the partitioned matrix according to rowID for each worker
			// i.e. (workerID => ordered list[(rowBlockID, (single row features, single row labels)]
			.aggregateByKey(new LinkedList<Tuple2<Long, Tuple2<MatrixBlock, MatrixBlock>>>(), new Partitioner() {
				private static final long serialVersionUID = -7937781374718031224L;
				@Override
				public int getPartition(Object workerID) {
					return (int) workerID;
				}
				@Override
				public int numPartitions() {
					return workerNum;
				}
			}, (list, input) -> {
				list.add(input);
				return list;
			}, (l1, l2) -> {
				l1.addAll(l2);
				l1.sort((o1, o2) -> o1._1.compareTo(o2._1));
				return l1;
			})
			.mapToPair(new DataPartitionerSparkAggregator(features.getNumColumns(), labels.getNumColumns()));

		if (DMLScript.STATISTICS)
			Statistics.accPSSetupTime((long) tSetup.stop());
		return result;
	}
}
