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

package org.apache.sysml.runtime.instructions.cp;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.sysml.parser.Expression;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.controlprogram.paramserv.LocalAggregationService;
import org.apache.sysml.runtime.controlprogram.paramserv.LocalPSWorker;
import org.apache.sysml.runtime.controlprogram.paramserv.LocalParamServer;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamServer;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysml.runtime.matrix.operators.Operator;

public class ParamservBuiltinCPInstruction extends ParameterizedBuiltinCPInstruction {

	protected ParamservBuiltinCPInstruction(Operator op, LinkedHashMap<String, String> paramsMap, CPOperand out,
			String opcode, String istr) {
		super(op, paramsMap, out, opcode, istr);
	}

	private CPOperand createOperand(String name, Expression.ValueType vt) {
		return new CPOperand(name, vt, Expression.DataType.SCALAR, true);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		// Determine the number of workers
		int workerNum = Integer.valueOf(getParam(Statement.PS_PARALLELISM));

		// Start the parameter server
		ParamServer ps = createPS();

		// Create aggregation service
		String aggFunc = getParam(Statement.PS_AGGREGATION_FUN);
		ListObject globalParams = ec.getListObject(getParam(Statement.PS_MODEL));
		ListObject hyperParams = getHyperParams(ec);
		LocalAggregationService aggService = new LocalAggregationService(aggFunc,
				ExecutionContextFactory.createContext(ec.getProgram()), globalParams, hyperParams, ps, workerNum);
		Thread aggThread = new Thread(aggService);

		// Create the local workers
		String updFunc = getParam(Statement.PS_UPDATE_FUN);
		Statement.PSFrequency freq = Statement.PSFrequency.valueOf(getParam(Statement.PS_FREQUENCY));
		int epochs = Integer.valueOf(getParam(Statement.PS_EPOCHS));
		if (epochs <= 0) {
			throw new DMLRuntimeException(
					String.format("Paramserv function: The argument '%s' could not be less than or equal to 0.",
							Statement.PS_EPOCHS));
		}
		long batchSize = getBatchSize(freq, ec);
		List<LocalPSWorker> workers = IntStream.range(0, workerNum).mapToObj(
				i -> new LocalPSWorker((long) i, updFunc, freq, epochs, batchSize, hyperParams,
						ExecutionContextFactory.createContext(ec.getProgram()), ps)).collect(Collectors.toList());

		// Do data partition
		doDataPartition(workerNum, ec, workers);

		// Create the worker threads
		List<Thread> threads = workers.stream().map(Thread::new).collect(Collectors.toList());

		// Start the threads
		threads.forEach(Thread::start);
		aggThread.start();

		// Wait for the threads stoping
		threads.forEach(thread -> {
			try {
				thread.join();
			} catch (InterruptedException e) {
				throw new DMLRuntimeException("Paramserv function: Failed to join the worker threads.", e);
			}
		});
		LOG.info("All workers finished.");
		aggService.kill();

		// Create the output
		ListObject result = (ListObject) ps.pull(ParamServer.RESULT_MODEL);
		ec.setVariable(output.getName(), result);
	}

	private ParamServer createPS() {
		ParamServer ps = null;
		final Statement.PSModeType mode = Statement.PSModeType.valueOf(getParam(Statement.PS_MODE));
		switch (mode) {
		case LOCAL:
			ps = new LocalParamServer();
			break;
		}
		return ps;
	}

	private long getBatchSize(Statement.PSFrequency freq, ExecutionContext ec) {
		long batchSize = Integer.valueOf(getParam(Statement.PS_BATCH_SIZE));
		if (batchSize <= 0) {
			throw new DMLRuntimeException(
					String.format("In paramserv function, the argument '%s' could not be less than or equal to 0.",
							Statement.PS_BATCH_SIZE));
		}
		if (freq.equals(Statement.PSFrequency.EPOCH)) {
			batchSize = ec.getMatrixObject(getParam(Statement.PS_FEATURES)).getNumRows();
		}
		return batchSize;
	}

	private ListObject getHyperParams(ExecutionContext ec) {
		ListObject hyperparams = null;
		if (getParameterMap().containsKey(Statement.PS_HYPER_PARAMS)) {
			hyperparams = ec.getListObject(getParam(Statement.PS_HYPER_PARAMS));
		}
		return hyperparams;
	}

	private void doDataPartition(int workerNum, ExecutionContext ec, List<LocalPSWorker> workers) {
		MatrixObject features = ec.getMatrixObject(getParam(Statement.PS_FEATURES));
		MatrixObject labels = ec.getMatrixObject(getParam(Statement.PS_LABELS));
		MatrixObject valFeatures = ec.getMatrixObject(getParam(Statement.PS_VAL_FEATURES));
		MatrixObject valLabels = ec.getMatrixObject(getParam(Statement.PS_VAL_LABELS));
		Statement.PSScheme scheme = Statement.PSScheme.valueOf(getParam(Statement.PS_SCHEME));
		switch (scheme) {
		case DISJOINT_CONTIGUOUS:
			disjointContiguous(workerNum, features, labels, valFeatures, valLabels, workers);
			break;
		}
	}

	private void disjointContiguous(int workerNum, MatrixObject features, MatrixObject labels, MatrixObject valFeatures,
			MatrixObject valLabels, List<LocalPSWorker> workers) {
		// training data
		List<MatrixObject> pfs = disjointContiguous(workerNum, features);
		List<MatrixObject> pls = disjointContiguous(workerNum, labels);
		for (int i = 0; i < workers.size(); i++) {
			workers.get(i).setFeatures(pfs.get(i));
			workers.get(i).setLabels(pls.get(i));
		}

		// validation data
		List<MatrixObject> pvfs = disjointContiguous(workerNum, valFeatures);
		List<MatrixObject> pvls = disjointContiguous(workerNum, valLabels);
		for (int i = 0; i < workers.size(); i++) {
			workers.get(i).setValFeatures(pvfs.get(i));
			workers.get(i).setValLabels(pvls.get(i));
		}
	}

	private List<MatrixObject> disjointContiguous(int workerNum, MatrixObject mo) {
		List<MatrixObject> list = new ArrayList<>();
		long stepSize = mo.getNumRows() / workerNum;
		long begin = 1;
		while (begin < mo.getNumRows()) {
			long end = begin + stepSize;
			if (end > mo.getNumRows()) {
				end = mo.getNumRows();
			}
			MatrixObject pmo = ParamservUtils.sliceMatrix(mo, begin, end);
			list.add(pmo);
			begin = end + 1;
		}
		return list;
	}
}
