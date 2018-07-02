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

package org.apache.sysml.test.integration.functions.paramserv;

import static org.apache.sysml.hops.OptimizerUtils.DEFAULT_BLOCKSIZE;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysml.runtime.controlprogram.paramserv.spark.DataPartitionerSparkMapper;
import org.apache.sysml.runtime.controlprogram.paramserv.spark.DataPartitionerSparkReducer;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.util.DataConverter;
import org.junit.Assert;
import org.junit.Test;

import scala.Tuple2;

public class SparkDataPartitionerTest {

	@Test
	public void testSparkDataPartitionerDC() {
		DataPartitionerSparkMapper mapper = new DataPartitionerSparkMapper(Statement.PSScheme.DISJOINT_CONTIGUOUS, 3);
		Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> result = doPartitioning(mapper);

		Assert.assertEquals(3, result.size());
		result.values().forEach(v -> {
			Assert.assertEquals(v._1.getNumRows(), v._2.getNumRows());
			double[] features = v._1.getDenseBlockValues();
			double[] labels = v._2.getDenseBlockValues();
			for (int i = 0; i < v._1.getNumRows(); i++) {
				for (int j = 0; j < v._1.getNumColumns(); j++) {
					Assert.assertEquals((int)labels[i], (int)features[i * v._1.getNumColumns() + j] / v._1.getNumColumns());
				}
			}
		});
	}

	@Test
	public void testSparkDataPartitionerDR() {
		DataPartitionerSparkMapper mapper = new DataPartitionerSparkMapper(Statement.PSScheme.DISJOINT_RANDOM, 4);
		Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> result = doPartitioning(mapper);

		List<MatrixBlock> pfs = result.values().stream().map(r -> r._1).collect(Collectors.toList());
		List<MatrixBlock> pls = result.values().stream().map(r -> r._2).collect(Collectors.toList());
		assertPermutationDR(4000 * 2000, pfs);
		assertPermutationDR(4000, pls);
	}

	@Test
	public void testSparkDataPartitionerDRR() {
		DataPartitionerSparkMapper mapper = new DataPartitionerSparkMapper(Statement.PSScheme.DISJOINT_ROUND_ROBIN, 4);
		Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> result = doPartitioning(mapper);

		Assert.assertEquals(4, result.size());
		result.forEach((key, value) -> {
			int workerID = key;
			MatrixBlock features = value._1;
			MatrixBlock labels = value._2;
			Assert.assertEquals(features.getNumRows(), labels.getNumRows());
			double[] fd = features.getDenseBlockValues();
			double[] ld = labels.getDenseBlockValues();
			for (int i = 0; i < features.getNumRows(); i++) {
				Assert.assertEquals(workerID, ((int) ld[i]) % result.size());
				for (int j = 0; j < labels.getNumColumns(); j++) {
					Assert.assertEquals((int) ld[i],
							(int) fd[i * features.getNumColumns() + j] / features.getNumColumns());
				}
			}
		});
	}


	@Test
	public void testSparkDataPartitionerOR() {
		DataPartitionerSparkMapper mapper = new DataPartitionerSparkMapper(Statement.PSScheme.OVERLAP_RESHUFFLE, 4);
		Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> result = doPartitioning(mapper);

		Assert.assertEquals(4, result.size());
		List<MatrixBlock> pfs = result.values().stream().map(r -> r._1).collect(Collectors.toList());
		List<MatrixBlock> pls = result.values().stream().map(r -> r._2).collect(Collectors.toList());
		assertPermutationOR(4000 * 2000, pfs);
		assertPermutationOR(4000, pls);
	}

	private Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> doPartitioning(DataPartitionerSparkMapper mapper) {
		double[][] df = new double[4000][2000];
		for (int i = 0; i < 4000; i++) {
			for (int j = 0; j < 2000; j++) {
				df[i][j] = i * 2000 + j;
			}
		}
		double[] dl = new double[4000];
		for (int i = 0; i < 4000; i++) {
			dl[i] = i;
		}
		MatrixBlock fmb = DataConverter.convertToMatrixBlock(df);
		MatrixBlock lmb = DataConverter.convertToMatrixBlock(dl, true);

		DataPartitionerSparkReducer reducer = new DataPartitionerSparkReducer();
		DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		JavaSparkContext jsc = SparkExecutionContext.getSparkContextStatic();
		JavaPairRDD<MatrixIndexes, MatrixBlock> featuresRDD = SparkExecutionContext
			.toMatrixJavaPairRDD(jsc, fmb, DEFAULT_BLOCKSIZE, DEFAULT_BLOCKSIZE);
		JavaPairRDD<MatrixIndexes, MatrixBlock> labelsRDD = SparkExecutionContext
			.toMatrixJavaPairRDD(jsc, lmb, DEFAULT_BLOCKSIZE, DEFAULT_BLOCKSIZE);

		return ParamservUtils.assembleTrainingData(featuresRDD, labelsRDD)
		    .flatMapToPair(mapper)   // Do the data partitioning on spark
		    .reduceByKey(reducer)    // Group partition and put them on each worker
		    .map((Function<Tuple2<Integer, Tuple2<MatrixBlock, MatrixBlock>>, Map<Integer, Tuple2<MatrixBlock, MatrixBlock>>>) input -> {
			   Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> tmp = new HashMap<>();
			   tmp.put(input._1, input._2);
			   return tmp;
		    })
		    .reduce((Function2<Map<Integer, Tuple2<MatrixBlock, MatrixBlock>>, Map<Integer, Tuple2<MatrixBlock, MatrixBlock>>, Map<Integer, Tuple2<MatrixBlock, MatrixBlock>>>)
				(o, o2) -> {
					o.putAll(o2);
					return o;
		    });
	}

	private void assertPermutationDR(int size, List<MatrixBlock> list) {
		Map<Double, Integer> dict = new HashMap<>();
		for (int i = 0; i < size; i++) {
			dict.put((double) i, 0);
		}
		IntStream.range(0, list.size()).forEach(i -> {
			double[] f = list.get(i).getDenseBlockValues();
			for (double d : f) {
				dict.compute(d, (k, v) -> v + 1);
			}
		});
		// check if all the occurrences are equivalent to one
		for (Map.Entry<Double, Integer> e : dict.entrySet()) {
			Assert.assertEquals(1, (int) e.getValue());
		}
	}

	private void assertPermutationOR(int size, List<MatrixBlock> list) {
		for (MatrixBlock mb : list) {
			Map<Double, Integer> dict = new HashMap<>();
			for (int i = 0; i < size; i++) {
				dict.put((double) i, 0);
			}
			double[] f = mb.getDenseBlockValues();
			for (double d : f) {
				dict.compute(d, (k, v) -> v + 1);
			}
			// check if all the occurrences are equivalent to one
			for (Map.Entry<Double, Integer> e : dict.entrySet()) {
				Assert.assertEquals(1, (int) e.getValue());
			}
		}
	}

}
