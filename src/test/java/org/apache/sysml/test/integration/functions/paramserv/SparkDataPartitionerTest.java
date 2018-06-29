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
		double[] df = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		MatrixBlock mb = DataConverter.convertToMatrixBlock(df, true);

		Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> result = doPartitioning(mapper, mb);
		Assert.assertEquals(3, result.size());
		double[] expected1 = new double[] { 1, 2, 3, 4 };
		assertResult(result, 0, expected1);
		double[] expected2 = new double[] { 5, 6, 7, 8 };
		assertResult(result, 1, expected2);
		double[] expected3 = new double[] { 9, 10 };
		assertResult(result, 2, expected3);
	}

	@Test
	public void testSparkDataPartitionerDR() {
		DataPartitionerSparkMapper mapper = new DataPartitionerSparkMapper(Statement.PSScheme.DISJOINT_RANDOM, 4);
		double[] df = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		MatrixBlock mb = DataConverter.convertToMatrixBlock(df, true);
		Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> result = doPartitioning(mapper, mb);

		List<MatrixBlock> pfs = result.values().stream().map(r -> r._1).collect(Collectors.toList());
		List<MatrixBlock> pls = result.values().stream().map(r -> r._2).collect(Collectors.toList());
		assertPermutationDR(df, pfs);
		assertPermutationDR(df, pls);
	}

	@Test
	public void testSparkDataPartitionerDRR() {
		DataPartitionerSparkMapper mapper = new DataPartitionerSparkMapper(Statement.PSScheme.DISJOINT_ROUND_ROBIN, 4);
		double[] df = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		MatrixBlock mb = DataConverter.convertToMatrixBlock(df, true);
		Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> result = doPartitioning(mapper, mb);

		Assert.assertEquals(4, result.size());
		double[] expected1 = new double[] { 4, 8 };
		assertResult(result, 0, expected1);
		double[] expected2 = new double[] { 1, 5, 9 };
		assertResult(result, 1, expected2);
		double[] expected3 = new double[] { 2, 6, 10 };
		assertResult(result, 2, expected3);
		double[] expected4 = new double[] { 3, 7 };
		assertResult(result, 3, expected4);
	}

	@Test
	public void testSparkDataPartitionerOR() {
		DataPartitionerSparkMapper mapper = new DataPartitionerSparkMapper(Statement.PSScheme.OVERLAP_RESHUFFLE, 4);
		double[] df = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		MatrixBlock mb = DataConverter.convertToMatrixBlock(df, true);
		Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> result = doPartitioning(mapper, mb);

		Assert.assertEquals(4, result.size());
		List<MatrixBlock> pfs = result.values().stream().map(r -> r._1).collect(Collectors.toList());
		List<MatrixBlock> pls = result.values().stream().map(r -> r._2).collect(Collectors.toList());
		assertPermutationOR(df, pfs);
		assertPermutationOR(df, pls);
	}

	private void assertResult(Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> result, int index, double[] expected) {
		double[] realValue1 = result.get(index)._1.getDenseBlockValues();
		double[] realValue2 = result.get(index)._2.getDenseBlockValues();
		Assert.assertArrayEquals(expected, realValue1, 0);
		Assert.assertArrayEquals(expected, realValue2, 0);
	}

	private Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> doPartitioning(DataPartitionerSparkMapper mapper, MatrixBlock mb) {
		DataPartitionerSparkReducer reducer = new DataPartitionerSparkReducer();
		DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		JavaSparkContext jsc = SparkExecutionContext.getSparkContextStatic();
		JavaPairRDD<MatrixIndexes, MatrixBlock> featuresRDD = SparkExecutionContext
			.toMatrixJavaPairRDD(jsc, mb, DEFAULT_BLOCKSIZE, DEFAULT_BLOCKSIZE);
		JavaPairRDD<MatrixIndexes, MatrixBlock> labelsRDD = SparkExecutionContext
			.toMatrixJavaPairRDD(jsc, mb, DEFAULT_BLOCKSIZE, DEFAULT_BLOCKSIZE);
		Object ro = featuresRDD
			.cogroup(labelsRDD)      // Combine RDDs of features and labels into a pair
		    .flatMapToPair(mapper)   // Do the data partitioning on spark
		    .reduceByKey(reducer)    // Group partition and put them on each worker
		    .map((Function<Tuple2<Integer, Tuple2<MatrixBlock, MatrixBlock>>, Object>) input -> {
			   Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> tmp = new HashMap<>();
			   tmp.put(input._1, input._2);
			   return tmp;
		    }).reduce((Function2<Object, Object, Object>) (o, o2) -> {
				Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> map1 = (Map<Integer, Tuple2<MatrixBlock, MatrixBlock>>) o;
				Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> map2 = (Map<Integer, Tuple2<MatrixBlock, MatrixBlock>>) o2;
				map1.putAll(map2);
				return map1;
		    });
		return (Map<Integer, Tuple2<MatrixBlock, MatrixBlock>>) ro;
	}

	private void assertPermutationDR(double[] df, List<MatrixBlock> list) {
		Map<Double, Integer> dict = new HashMap<>();
		for (double d : df) {
			dict.put(d, 0);
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

	private void assertPermutationOR(double[] df, List<MatrixBlock> list) {
		for (MatrixBlock mb : list) {
			Map<Double, Integer> dict = new HashMap<>();
			for (double d : df) {
				dict.put(d, 0);
			}
			double[] f = mb.getDenseBlockValues();
			for (double d : f) {
				dict.compute(d, (k, v) -> v + 1);
			}
			Assert.assertEquals(10, dict.size());
			// check if all the occurrences are equivalent to one
			for (Map.Entry<Double, Integer> e : dict.entrySet()) {
				Assert.assertEquals(1, (int) e.getValue());
			}
		}
	}

}
