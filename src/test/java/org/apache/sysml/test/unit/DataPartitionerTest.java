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

package org.apache.sysml.test.unit;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.controlprogram.paramserv.DataPartitioner;
import org.apache.sysml.runtime.controlprogram.paramserv.DataPartitionerDC;
import org.apache.sysml.runtime.controlprogram.paramserv.DataPartitionerDR;
import org.apache.sysml.runtime.controlprogram.paramserv.DataPartitionerDRR;
import org.apache.sysml.runtime.controlprogram.paramserv.DataPartitionerOR;
import org.apache.sysml.runtime.controlprogram.paramserv.LocalPSWorker;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;
import org.junit.Assert;
import org.junit.Test;

public class DataPartitionerTest {

	@Test
	public void testDataPartitionerDC() {
		DataPartitioner dp = new DataPartitionerDC();
		List<LocalPSWorker> workers = IntStream.range(0, 2).mapToObj(i -> new LocalPSWorker(i, "updFunc", Statement.PSFrequency.BATCH, 1, 64, null, null, createMockExecutionContext(), null)).collect(Collectors.toList());
		double[] df = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		MatrixObject features = ParamservUtils.newMatrixObject();
		features.acquireModify(DataConverter.convertToMatrixBlock(df, true));
		features.refreshMetaData();
		features.release();
		MatrixObject labels = ParamservUtils.newMatrixObject();
		labels.acquireModify(DataConverter.convertToMatrixBlock(df, true));
		labels.refreshMetaData();
		labels.release();
		dp.doPartitioning(workers, features, labels);

		double[] expected1 = new double[] { 1, 2, 3, 4, 5 };
		double[] realValue1 = workers.get(0).getFeatures().acquireRead().getDenseBlockValues();
		double[] realValue2 = workers.get(0).getLabels().acquireRead().getDenseBlockValues();
		Assert.assertArrayEquals(expected1, realValue1, 0);
		Assert.assertArrayEquals(expected1, realValue2, 0);

		double[] expected2 = new double[] { 6, 7, 8, 9, 10 };
		double[] realValue3 = workers.get(1).getFeatures().acquireRead().getDenseBlockValues();
		double[] realValue4 = workers.get(1).getLabels().acquireRead().getDenseBlockValues();
		Assert.assertArrayEquals(expected2, realValue3, 0);
		Assert.assertArrayEquals(expected2, realValue4, 0);
	}

	@Test
	public void testDataPartitionerDR() {
		DataPartitionerDR dp = new DataPartitionerDR();
		double[] df = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		MatrixObject features = ParamservUtils.newMatrixObject();
		features.acquireModify(DataConverter.convertToMatrixBlock(df, true));
		features.refreshMetaData();
		features.release();
		MatrixObject labels = ParamservUtils.newMatrixObject();
		labels.acquireModify(DataConverter.convertToMatrixBlock(df, true));
		labels.refreshMetaData();
		labels.release();

		MatrixBlock permutation = ParamservUtils.generatePermutation(df.length, df.length);

		List<MatrixObject> pfs = dp.doPartitioning(2, features, permutation);
		List<MatrixObject> pls = dp.doPartitioning(2, labels, permutation);

		double[] expected1 = IntStream.range(0, 5).mapToDouble(i -> permutation.getSparseBlock().get(i).indexes()[0] + 1).toArray();
		double[] realValue1 = pfs.get(0).acquireRead().getDenseBlockValues();
		double[] realValue2 = pls.get(0).acquireRead().getDenseBlockValues();
		Assert.assertArrayEquals(expected1, realValue1, 0);
		Assert.assertArrayEquals(expected1, realValue2, 0);

		double[] expected2 = IntStream.range(5, 10).mapToDouble(i -> permutation.getSparseBlock().get(i).indexes()[0] + 1).toArray();
		double[] realValue3 = pfs.get(1).acquireRead().getDenseBlockValues();
		double[] realValue4 = pls.get(1).acquireRead().getDenseBlockValues();
		Assert.assertArrayEquals(expected2, realValue3, 0);
		Assert.assertArrayEquals(expected2, realValue4, 0);
	}

	@Test
	public void testDataPartitionerDRR() {
		DataPartitioner dp = new DataPartitionerDRR();
		List<LocalPSWorker> workers = IntStream.range(0, 2).mapToObj(i -> new LocalPSWorker(i, "updFunc", Statement.PSFrequency.BATCH, 1, 64, null, null, createMockExecutionContext(), null)).collect(Collectors.toList());
		double[] df = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		MatrixObject features = ParamservUtils.newMatrixObject();
		features.acquireModify(DataConverter.convertToMatrixBlock(df, true));
		features.refreshMetaData();
		features.release();
		MatrixObject labels = ParamservUtils.newMatrixObject();
		labels.acquireModify(DataConverter.convertToMatrixBlock(df, true));
		labels.refreshMetaData();
		labels.release();
		dp.doPartitioning(workers, features, labels);

		double[] expected1 = new double[] { 2, 4, 6, 8, 10 };
		double[] realValue1 = workers.get(0).getFeatures().acquireRead().getDenseBlockValues();
		double[] realValue2 = workers.get(0).getLabels().acquireRead().getDenseBlockValues();
		Assert.assertArrayEquals(expected1, realValue1, 0);
		Assert.assertArrayEquals(expected1, realValue2, 0);

		double[] expected2 = new double[] { 1, 3, 5, 7, 9 };
		double[] realValue3 = workers.get(1).getFeatures().acquireRead().getDenseBlockValues();
		double[] realValue4 = workers.get(1).getLabels().acquireRead().getDenseBlockValues();
		Assert.assertArrayEquals(expected2, realValue3, 0);
		Assert.assertArrayEquals(expected2, realValue4, 0);
	}

	@Test
	public void testDataPartitionerOR() {
		DataPartitionerOR dp = new DataPartitionerOR();
		double[] df = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		MatrixObject features = ParamservUtils.newMatrixObject();
		features.acquireModify(DataConverter.convertToMatrixBlock(df, true));
		features.refreshMetaData();
		features.release();
		MatrixObject labels = ParamservUtils.newMatrixObject();
		labels.acquireModify(DataConverter.convertToMatrixBlock(df, true));
		labels.refreshMetaData();
		labels.release();

		MatrixBlock permutation = ParamservUtils.generatePermutation(df.length, df.length);

		List<MatrixObject> pfs = dp.doPartitioning(1, features, permutation);
		List<MatrixObject> pls = dp.doPartitioning(1, labels, permutation);

		double[] expected1 = IntStream.range(0, 10).mapToDouble(i -> permutation.getSparseBlock().get(i).indexes()[0] + 1).toArray();
		double[] realValue1 = pfs.get(0).acquireRead().getDenseBlockValues();
		double[] realValue2 = pls.get(0).acquireRead().getDenseBlockValues();
		Assert.assertArrayEquals(expected1, realValue1, 0);
		Assert.assertArrayEquals(expected1, realValue2, 0);
	}

	private ExecutionContext createMockExecutionContext() {
		Program prog = new Program();
		ArrayList<DataIdentifier> inputs = new ArrayList<>();
		DataIdentifier features = new DataIdentifier("features");
		features.setDataType(Expression.DataType.MATRIX);
		features.setValueType(Expression.ValueType.DOUBLE);
		inputs.add(features);
		DataIdentifier labels = new DataIdentifier("labels");
		labels.setDataType(Expression.DataType.MATRIX);
		labels.setValueType(Expression.ValueType.DOUBLE);
		inputs.add(labels);
		DataIdentifier model = new DataIdentifier("model");
		model.setDataType(Expression.DataType.LIST);
		model.setValueType(Expression.ValueType.UNKNOWN);
		inputs.add(model);

		ArrayList<DataIdentifier> outputs = new ArrayList<>();
		DataIdentifier gradients = new DataIdentifier("gradients");
		gradients.setDataType(Expression.DataType.LIST);
		gradients.setValueType(Expression.ValueType.UNKNOWN);
		outputs.add(gradients);

		FunctionProgramBlock fpb = new FunctionProgramBlock(prog, inputs, outputs);
		prog.addProgramBlock(fpb);
		prog.addFunctionProgramBlock(DMLProgram.DEFAULT_NAMESPACE, "updFunc", fpb);
		return ExecutionContextFactory.createContext(prog);
	}
}
