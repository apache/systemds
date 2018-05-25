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
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.utils.NativeHelper;

public class ParamservBuiltinCPInstruction extends ParameterizedBuiltinCPInstruction {

	protected ParamservBuiltinCPInstruction(Operator op, LinkedHashMap<String, String> paramsMap, CPOperand out,
			String opcode, String istr) {
		super(op, paramsMap, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		// Determine the number of workers
		int workerNum = getWorkerNum();

		// Start the parameter server
		ParamServer ps = createPS();

		String aggFunc = getParam(Statement.PS_AGGREGATION_FUN);
		ListObject globalParams = ec.getListObject(getParam(Statement.PS_MODEL));
		ListObject hyperParams = getHyperParams(ec);

		// Create aggregation service
		LocalAggregationService aggService = new LocalAggregationService(aggFunc,
				ExecutionContextFactory.createContext(ec.getProgram()), ParamservUtils.copyList(globalParams),
				ParamservUtils.copyList(hyperParams), ps, workerNum);

		Thread aggThread = new Thread(aggService);

		String updFunc = getParam(Statement.PS_UPDATE_FUN);
		Statement.PSFrequency freq = Statement.PSFrequency.valueOf(getParam(Statement.PS_FREQUENCY));
		int epochs = Integer.valueOf(getParam(Statement.PS_EPOCHS));
		if (epochs <= 0) {
			throw new DMLRuntimeException(
					String.format("Paramserv function: The argument '%s' could not be less than or equal to 0.",
							Statement.PS_EPOCHS));
		}
		long batchSize = getBatchSize(freq, ec);

		// Create the local workers
		List<LocalPSWorker> workers = IntStream.range(0, workerNum).mapToObj(
				i -> new LocalPSWorker((long) i, updFunc, freq, epochs, batchSize, ParamservUtils.copyList(hyperParams),
						ExecutionContextFactory.createContext(ec.getProgram()), ps)).collect(Collectors.toList());

		// Do data partition
		doDataPartition(workerNum, ec, workers);

		// Create the worker threads
		List<Thread> threads = workers.stream().map(Thread::new).collect(Collectors.toList());

		// Start the threads
		threads.forEach(Thread::start);
		aggThread.start();

		// Wait for the threads stopping
		threads.forEach(thread -> {
			try {
				thread.join();
			} catch (InterruptedException e) {
				throw new DMLRuntimeException("Paramserv function: Failed to join the worker threads.", e);
			}
		});
		LOG.info("All workers finished.");

		aggService.kill();
		try {
			aggThread.join();
		} catch (InterruptedException e) {
			throw new DMLRuntimeException("Paramserv function: Failed to join the aggregation service thread.", e);
		}
		LOG.info("Aggregation service finished.");

		// Create the output
		ListObject result = (ListObject) ps.pull(ParamServer.RESULT_MODEL);
		result.setStatus(ec.pinVariables(result.getNames()));
		ec.setVariable(output.getName(), result);

		ParamservUtils.cleanupListObject(ec, globalParams);
		ParamservUtils.cleanupListObject(ec, hyperParams);

		ps.getParams().clear();
	}

	/**
	 * Get the worker numbers according to the vcores
	 *
	 * @return worker numbers
	 */
	private int getWorkerNum() {
		int workerNum = Integer.valueOf(getParam(Statement.PS_PARALLELISM));
		Statement.PSModeType mode = Statement.PSModeType.valueOf(getParam(Statement.PS_MODE));
		switch (mode) {
		case LOCAL:
			int vcores = InfrastructureAnalyzer.getLocalParallelism();
			if ("openblas".equals(NativeHelper.getCurrentBLAS())) {
				workerNum = Math.min(workerNum, vcores / 2);
			} else {
				workerNum = Math.min(workerNum, vcores);
			}
			break;
		case REMOTE_SPARK:
			throw new DMLRuntimeException("Do not support remote spark.");
		}
		return workerNum;
	}

	/**
	 * Create a server which serves the local or remote workers
	 *
	 * @return parameter server
	 */
	private ParamServer createPS() {
		ParamServer ps = null;
		Statement.PSModeType mode = Statement.PSModeType.valueOf(getParam(Statement.PS_MODE));
		switch (mode) {
		case LOCAL:
			ps = new LocalParamServer();
			break;
		case REMOTE_SPARK:
			throw new DMLRuntimeException("Do not support remote spark.");
		}
		return ps;
	}

	private long getBatchSize(Statement.PSFrequency freq, ExecutionContext ec) {
		long batchSize = Integer.valueOf(getParam(Statement.PS_BATCH_SIZE));
		if (batchSize <= 0) {
			throw new DMLRuntimeException(String.format(
					"Paramserv function: the number of argument '%s' could not be less than or equal to 0.",
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
		case DISJOINT_RANDOM:
		case OVERLAP_RESHUFFLE:
		case DISJOINT_ROUND_ROBIN:
			throw new DMLRuntimeException(
					String.format("Paramserv function: the scheme '%s' is not supported.", scheme));
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
