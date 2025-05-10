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

package org.apache.sysds.test.functions.paramserv;

import java.util.Map;
import java.util.stream.IntStream;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysds.runtime.controlprogram.paramserv.SparkParamservUtils;
import org.apache.sysds.runtime.controlprogram.paramserv.dp.DataPartitionLocalScheme;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Assert;
import org.junit.Test;

import scala.Tuple2;

@net.jcip.annotations.NotThreadSafe
public class SparkDataPartitionerTest extends BaseDataPartitionerTest {

	private static SparkExecutionContext _sec;

	static {
		DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		DMLScript.setGlobalExecMode(ExecMode.SPARK);
		_sec =  ExecutionContextFactory.createSparkExecutionContext();
	}

	private Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> doPartitioning(Statement.PSScheme scheme) {
		MatrixBlock[] mbs = generateData();
		return SparkParamservUtils.doPartitionOnSpark(_sec,
			ParamservUtils.newMatrixObject(mbs[0], false), 
			ParamservUtils.newMatrixObject(mbs[1]), scheme, WORKER_NUM).collectAsMap();
	}

	@Test
	public void testSparkDataPartitionerDC() {
		DataPartitionLocalScheme.Result localResult = launchLocalDataPartitionerDC();
		Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> sparkResult = doPartitioning(Statement.PSScheme.DISJOINT_CONTIGUOUS);

		// Compare the both
		assertResult(localResult, sparkResult);
	}

	private static void assertResult(DataPartitionLocalScheme.Result local, Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> spark) {
		IntStream.range(0, WORKER_NUM).forEach(w -> {
			Assert.assertArrayEquals(local.pFeatures.get(w).acquireRead().getDenseBlockValues(), spark.get(w)._1.getDenseBlockValues(), 0);
			Assert.assertArrayEquals(local.pLabels.get(w).acquireRead().getDenseBlockValues(), spark.get(w)._2.getDenseBlockValues(), 0);
		});
	}

	@Test
	public void testSparkDataPartitionerDR() {
		ParamservUtils.SEED = System.nanoTime();
		MatrixBlock[] mbs = generateData();
		DataPartitionLocalScheme.Result localResult = launchLocalDataPartitionerDR(mbs);
		Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> sparkResult = doPartitioning(Statement.PSScheme.DISJOINT_RANDOM);

		// Compare the both
		assertResult(localResult, sparkResult);
	}

	@Test
	public void testSparkDataPartitionerDRR() {
		DataPartitionLocalScheme.Result localResult = launchLocalDataPartitionerDRR();
		Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> sparkResult = doPartitioning(Statement.PSScheme.DISJOINT_ROUND_ROBIN);

		// Compare the both
		assertResult(localResult, sparkResult);
	}

	@Test
	public void testSparkDataPartitionerOR() {
		ParamservUtils.SEED = System.nanoTime();
		DataPartitionLocalScheme.Result localResult = launchLocalDataPartitionerOR();
		Map<Integer, Tuple2<MatrixBlock, MatrixBlock>> sparkResult = doPartitioning(Statement.PSScheme.OVERLAP_RESHUFFLE);

		// Compare the both
		assertResult(localResult, sparkResult);
	}
}
