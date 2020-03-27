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

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysds.runtime.controlprogram.paramserv.dp.DataPartitionLocalScheme;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import scala.Tuple2;

public class LocalDataPartitionerTest extends BaseDataPartitionerTest {

	@Test
	public void testLocalDataPartitionerDC() {
		DataPartitionLocalScheme.Result result = launchLocalDataPartitionerDC();

		Assert.assertEquals(WORKER_NUM, result.pFeatures.size());
		Assert.assertEquals(WORKER_NUM, result.pLabels.size());
		for (int i = 0; i < WORKER_NUM; i++) {
			assertDCResult(result, i);
		}
	}

	private void assertDCResult(DataPartitionLocalScheme.Result result, int workerID) {
		Assert.assertArrayEquals(generateExpectedData(workerID * (ROW_SIZE / WORKER_NUM) * COL_SIZE, (workerID + 1) * (ROW_SIZE / WORKER_NUM) * COL_SIZE), result.pFeatures.get(workerID).acquireRead().getDenseBlockValues(), 0);
		Assert.assertArrayEquals(generateExpectedData(workerID * (ROW_SIZE / WORKER_NUM), (workerID + 1) * (ROW_SIZE / WORKER_NUM)), result.pLabels.get(workerID).acquireRead().getDenseBlockValues(), 0);
	}

	@Test
	public void testLocalDataPartitionerDR() {
		MatrixBlock[] mbs = generateData();
		DataPartitionLocalScheme.Result result = launchLocalDataPartitionerDR(mbs);

		Assert.assertEquals(WORKER_NUM, result.pFeatures.size());
		Assert.assertEquals(WORKER_NUM, result.pLabels.size());

		// Generate the expected data
		MatrixBlock perm = ParamservUtils.generatePermutation(ROW_SIZE, ParamservUtils.SEED);
		List<MatrixBlock> efs = generateExpectedDataDR(mbs[0], perm);
		List<MatrixBlock> els = generateExpectedDataDR(mbs[1], perm);

		for (int i = 0; i < WORKER_NUM; i++) {
			Assert.assertArrayEquals(efs.get(i).getDenseBlockValues(), result.pFeatures.get(i).acquireRead().getDenseBlockValues(), 0);
			Assert.assertArrayEquals(els.get(i).getDenseBlockValues(), result.pLabels.get(i).acquireRead().getDenseBlockValues(), 0);
		}
	}

	private static List<MatrixBlock> generateExpectedDataDR(MatrixBlock mb, MatrixBlock perm) {
		int batchSize = (int) Math.ceil((double) ROW_SIZE / WORKER_NUM);
		return IntStream.range(0, WORKER_NUM).mapToObj(i -> {
			int begin = i * batchSize;
			int end = Math.min((i + 1) * batchSize, mb.getNumRows());
			MatrixBlock slicedPerm = perm.slice(begin, end - 1);
			return slicedPerm.aggregateBinaryOperations(slicedPerm, mb, new MatrixBlock(),
					InstructionUtils.getMatMultOperator(WORKER_NUM));
		}).collect(Collectors.toList());
	}

	@Test
	public void testLocalDataPartitionerDRR() {
		DataPartitionLocalScheme.Result result = launchLocalDataPartitionerDRR();

		Assert.assertEquals(WORKER_NUM, result.pFeatures.size());
		Assert.assertEquals(WORKER_NUM, result.pLabels.size());
		for (int i = 0; i < WORKER_NUM; i++) {
			assertDRRResult(result, i);
		}
	}

	private static void assertDRRResult(DataPartitionLocalScheme.Result result, int workerID) {
		Tuple2<double[], double[]> expected = generateExpectedData(workerID, WORKER_NUM, ROW_SIZE / WORKER_NUM);
		Assert.assertArrayEquals(expected._1, result.pFeatures.get(workerID).acquireRead().getDenseBlockValues(), 0);
		Assert.assertArrayEquals(expected._2, result.pLabels.get(workerID).acquireRead().getDenseBlockValues(), 0);
	}

	private static Tuple2<double[], double[]> generateExpectedData(int start, int step, int rowSize) {
		double[] features = new double[rowSize * COL_SIZE];
		int fIndex = 0;
		double[] labels = new double[rowSize];
		for (int i = 0; i < rowSize; i++) {
			int rowID = start + i * step;
			labels[i] = rowID;
			for (int j = rowID * COL_SIZE; j < (rowID + 1) * COL_SIZE; j++) {
				features[fIndex++] = j;
			}
		}
		return new Tuple2<>(features, labels);
	}

	@Test
	public void testLocalDataPartitionerOR() {
		ParamservUtils.SEED = System.nanoTime();
		DataPartitionLocalScheme.Result result = launchLocalDataPartitionerOR();

		Assert.assertEquals(WORKER_NUM, result.pFeatures.size());
		Assert.assertEquals(WORKER_NUM, result.pLabels.size());
		for (int i = 0; i < WORKER_NUM; i++) {
			Tuple2<MatrixBlock, MatrixBlock> expected = generateExpectedDataOR(i);
			Assert.assertArrayEquals(expected._1.getDenseBlockValues(), result.pFeatures.get(i).acquireRead().getDenseBlockValues(), 0);
			Assert.assertArrayEquals(expected._2.getDenseBlockValues(), result.pLabels.get(i).acquireRead().getDenseBlockValues(), 0);
		}
	}

	private Tuple2<MatrixBlock, MatrixBlock> generateExpectedDataOR(int workerID) {
		MatrixBlock[] mbs = generateData();
		MatrixBlock perm = ParamservUtils.generatePermutation(ROW_SIZE, ParamservUtils.SEED+workerID);
		return new Tuple2<>(perm.aggregateBinaryOperations(perm, mbs[0], new MatrixBlock(), InstructionUtils.getMatMultOperator(WORKER_NUM)),
			perm.aggregateBinaryOperations(perm, mbs[1], new MatrixBlock(), InstructionUtils.getMatMultOperator(WORKER_NUM)));
	}

}
