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

import static org.apache.sysml.parser.Statement.PSFrequency;
import static org.apache.sysml.parser.Statement.PSModeType;
import static org.apache.sysml.parser.Statement.PSScheme;
import static org.apache.sysml.parser.Statement.PS_AGGREGATION_FUN;
import static org.apache.sysml.parser.Statement.PS_BATCH_SIZE;
import static org.apache.sysml.parser.Statement.PS_EPOCHS;
import static org.apache.sysml.parser.Statement.PS_FEATURES;
import static org.apache.sysml.parser.Statement.PS_FREQUENCY;
import static org.apache.sysml.parser.Statement.PS_HYPER_PARAMS;
import static org.apache.sysml.parser.Statement.PS_LABELS;
import static org.apache.sysml.parser.Statement.PS_MODE;
import static org.apache.sysml.parser.Statement.PS_MODEL;
import static org.apache.sysml.parser.Statement.PS_PARALLELISM;
import static org.apache.sysml.parser.Statement.PS_SCHEME;
import static org.apache.sysml.parser.Statement.PS_UPDATE_FUN;
import static org.apache.sysml.parser.Statement.PS_VAL_FEATURES;
import static org.apache.sysml.parser.Statement.PS_VAL_LABELS;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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

	private static final int DEFAULT_BATCH_SIZE = 64;
	private static final PSFrequency DEFAULT_UPDATE_FREQUENCY = PSFrequency.BATCH;
	private static final int DEFAULT_LEVEL_PARALLELISM = InfrastructureAnalyzer.getLocalParallelism();
	private static final PSScheme DEFAULT_SCHEME = PSScheme.DISJOINT_CONTIGUOUS;

	private int _workerNum;
	private PSModeType _mode;
	private String _updFunc;
	private String _aggFunc;
	private PSFrequency _freq;
	private int _epochs;
	private long _batchSize;

	protected ParamservBuiltinCPInstruction(Operator op, LinkedHashMap<String, String> paramsMap, CPOperand out,
			String opcode, String istr) {
		super(op, paramsMap, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {

		init();

		// Start the parameter server
		ParamServer ps = createPS();

		// Create aggregation service
		ListObject globalParams = ec.getListObject(getParam(PS_MODEL));
		ListObject hyperParams = getHyperParams(ec);
		LocalAggregationService aggService = new LocalAggregationService(_aggFunc,
				ExecutionContextFactory.createContext(ec.getProgram()), ParamservUtils.copyList(globalParams),
				ParamservUtils.copyList(hyperParams), ps, _workerNum);

		Thread aggThread = new Thread(aggService);

		// Create the local workers
		List<LocalPSWorker> workers = IntStream.range(0, _workerNum).mapToObj(
				i -> new LocalPSWorker((long) i, _updFunc, _freq, _epochs, _batchSize,
						ParamservUtils.copyList(hyperParams), ExecutionContextFactory.createContext(ec.getProgram()),
						ps)).collect(Collectors.toList());

		// Do data partition
		doDataPartition(ec, workers);

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

	private void init() {
		_mode = PSModeType.valueOf(getParam(PS_MODE));
		_workerNum = getWorkerNum();
		_updFunc = getParam(PS_UPDATE_FUN);
		_aggFunc = getParam(PS_AGGREGATION_FUN);
		_freq = getFrequency();
		_epochs = Integer.valueOf(getParam(PS_EPOCHS));
		if (_epochs <= 0) {
			throw new DMLRuntimeException(
					String.format("Paramserv function: The argument '%s' could not be less than or equal to 0.",
							PS_EPOCHS));
		}
		_batchSize = getBatchSize();
	}

	private PSFrequency getFrequency() {
		if (!getParameterMap().containsKey(PS_FREQUENCY)) {
			return DEFAULT_UPDATE_FREQUENCY; // default updating frequency
		}
		return PSFrequency.valueOf(getParam(PS_FREQUENCY));
	}

	/**
	 * Get the worker numbers according to the vcores
	 *
	 * @return worker numbers
	 */
	private int getWorkerNum() {
		int workerNum = DEFAULT_LEVEL_PARALLELISM;
		if (getParameterMap().containsKey(PS_PARALLELISM)) {
			workerNum = Integer.valueOf(getParam(PS_PARALLELISM));
		}
		switch (_mode) {
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
		switch (_mode) {
		case LOCAL:
			ps = new LocalParamServer();
			break;
		case REMOTE_SPARK:
			throw new DMLRuntimeException("Do not support remote spark.");
		}
		return ps;
	}

	private long getBatchSize() {
		if (!getParameterMap().containsKey(PS_BATCH_SIZE)) {
			return DEFAULT_BATCH_SIZE;
		}
		long batchSize = Integer.valueOf(getParam(PS_BATCH_SIZE));
		if (batchSize <= 0) {
			throw new DMLRuntimeException(String.format(
					"Paramserv function: the number of argument '%s' could not be less than or equal to 0.",
					PS_BATCH_SIZE));
		}
		return batchSize;
	}

	private ListObject getHyperParams(ExecutionContext ec) {
		ListObject hyperparams = new ListObject(Collections.emptyList());
		if (getParameterMap().containsKey(PS_HYPER_PARAMS)) {
			hyperparams = ec.getListObject(getParam(PS_HYPER_PARAMS));
		}
		return hyperparams;
	}

	private void doDataPartition(ExecutionContext ec, List<LocalPSWorker> workers) {
		MatrixObject features = ec.getMatrixObject(getParam(PS_FEATURES));
		MatrixObject labels = ec.getMatrixObject(getParam(PS_LABELS));
		MatrixObject valFeatures = ec.getMatrixObject(getParam(PS_VAL_FEATURES));
		MatrixObject valLabels = ec.getMatrixObject(getParam(PS_VAL_LABELS));
		PSScheme scheme = DEFAULT_SCHEME;
		if (getParameterMap().containsKey(PS_SCHEME)) {
			scheme = PSScheme.valueOf(getParam(PS_SCHEME));
		}
		switch (scheme) {
		case DISJOINT_CONTIGUOUS:
			disjointContiguous(features, labels, valFeatures, valLabels, workers);
			break;
		case DISJOINT_RANDOM:
		case OVERLAP_RESHUFFLE:
		case DISJOINT_ROUND_ROBIN:
			throw new DMLRuntimeException(
					String.format("Paramserv function: the scheme '%s' is not supported.", scheme));
		}
	}

	private void disjointContiguous(MatrixObject features, MatrixObject labels, MatrixObject valFeatures,
			MatrixObject valLabels, List<LocalPSWorker> workers) {
		// training data
		List<MatrixObject> pfs = disjointContiguous(features);
		List<MatrixObject> pls = disjointContiguous(labels);
		for (int i = 0; i < workers.size(); i++) {
			workers.get(i).setFeatures(pfs.get(i));
			workers.get(i).setLabels(pls.get(i));
		}

		// validation data
		List<MatrixObject> pvfs = disjointContiguous(valFeatures);
		List<MatrixObject> pvls = disjointContiguous(valLabels);
		for (int i = 0; i < workers.size(); i++) {
			workers.get(i).setValFeatures(pvfs.get(i));
			workers.get(i).setValLabels(pvls.get(i));
		}
	}

	private List<MatrixObject> disjointContiguous(MatrixObject mo) {
		List<MatrixObject> list = new ArrayList<>();
		long stepSize = mo.getNumRows() / _workerNum;
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
