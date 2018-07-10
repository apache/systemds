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

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.junit.Assert;
import org.junit.Test;

import scala.Tuple2;

public class SparkDataPartitionerTest extends BaseDataPartitionerTest implements Serializable {

	private static final long serialVersionUID = 3124325166708845429L;
	private static SparkExecutionContext _sec;

	static {
		DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		DMLScript.rtplatform = DMLScript.RUNTIME_PLATFORM.SPARK;
		_sec = (SparkExecutionContext) ExecutionContextFactory.createContext(null);
	}

	@Test
	public void testSparkDataPartitionerDC() {
		Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> result = doPartitioning(Statement.PSScheme.DISJOINT_CONTIGUOUS);
		Assert.assertEquals(WORKER_NUM, result.size());
		assertResult(result);
	}

	@Test
	public void testSparkDataPartitionerDR() {
		Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> result = doPartitioning(Statement.PSScheme.DISJOINT_RANDOM);
		Assert.assertEquals(WORKER_NUM, result.size());
		assertResult(result);
	}

	@Test
	public void testSparkDataPartitionerDRR() {
		Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> result = doPartitioning(Statement.PSScheme.DISJOINT_ROUND_ROBIN);

		Assert.assertEquals(WORKER_NUM, result.size());
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
					Assert.assertEquals((int) ld[i], (int) fd[i * features.getNumColumns() + j] / features.getNumColumns());
				}
			}
		});
	}

	@Test
	public void testSparkDataPartitionerOR() {
		Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> result = doPartitioning(Statement.PSScheme.OVERLAP_RESHUFFLE);

		Assert.assertEquals(WORKER_NUM, result.size());
		List<MatrixBlock> pfs = result.values().stream().map(r -> r._1).collect(Collectors.toList());
		List<MatrixBlock> pls = result.values().stream().map(r -> r._2).collect(Collectors.toList());
		assertPermutationOR(ROW_SIZE * COL_SIZE, pfs);
		assertPermutationOR(ROW_SIZE, pls);
	}

	private Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> doPartitioning(Statement.PSScheme scheme) {
		MatrixBlock[] mbs = generateData();
		return ParamservUtils.doPartitionOnSpark(_sec, ParamservUtils.newMatrixObject(mbs[0]), ParamservUtils.newMatrixObject(mbs[1]), scheme, WORKER_NUM)
				  .collectAsMap();
	}

	private void assertResult(List<MatrixBlock> list) {
		Map<Double, Integer> dict = new HashMap<>();
		for (MatrixBlock mb : list) {
			double[] dArray = mb.getDenseBlockValues();
			for (double d : dArray) {
				if (dict.containsKey(d)) {
					dict.compute(d, (aDouble, occ) -> occ + 1);
				} else {
					dict.put(d, 1);
				}
			}
		}
		for (int i = 1; i < ROW_SIZE; i++) {
			Assert.assertEquals(1, dict.get((double) i), 0);
		}
	}

	private void assertResult(Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> result) {
		List<MatrixBlock> pfs = result.values().stream().map(r -> r._1).collect(Collectors.toList());
		List<MatrixBlock> pls = result.values().stream().map(r -> r._2).collect(Collectors.toList());
		assertResult(pfs);
		assertResult(pls);
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
