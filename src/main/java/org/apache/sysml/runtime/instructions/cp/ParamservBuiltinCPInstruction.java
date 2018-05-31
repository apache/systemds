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
import static org.apache.sysml.parser.Statement.PSUpdateType;
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
import static org.apache.sysml.parser.Statement.PS_UPDATE_TYPE;
import static org.apache.sysml.parser.Statement.PS_VAL_FEATURES;
import static org.apache.sysml.parser.Statement.PS_VAL_LABELS;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
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

	//internal local debug level
	private static final boolean LDEBUG = false;

	private ExecutorService _ec;
	public static final int TIMEOUT = 10;

	static {
		// for internal debugging only
		if (LDEBUG) {
			Logger.getLogger("org.apache.sysml.runtime.controlprogram.paramserv").setLevel(Level.DEBUG);
		}
	}

	protected ParamservBuiltinCPInstruction(Operator op, LinkedHashMap<String, String> paramsMap, CPOperand out,
			String opcode, String istr) {
		super(op, paramsMap, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {

		PSModeType mode = PSModeType.valueOf(getParam(PS_MODE));
		int workerNum = getWorkerNum(mode);
		_ec = Executors.newFixedThreadPool(workerNum);
		String updFunc = getParam(PS_UPDATE_FUN);
		String aggFunc = getParam(PS_AGGREGATION_FUN);
		PSFrequency freq = getFrequency();
		PSUpdateType updateType = getUpdateType();
		int epochs = Integer.valueOf(getParam(PS_EPOCHS));
		if (epochs <= 0) {
			throw new DMLRuntimeException(
					String.format("Paramserv function: The argument '%s' could not be less than or equal to 0.",
							PS_EPOCHS));
		}
		long batchSize = getBatchSize();

		// Create the parameter server
		ListObject model = ec.getListObject(getParam(PS_MODEL));
		ListObject hyperParams = getHyperParams(ec);
		ParamServer ps = createPS(mode, aggFunc, freq, updateType, workerNum, model, ec, hyperParams);

		// Create the local workers
		List<LocalPSWorker> workers = IntStream.range(0, workerNum)
				.mapToObj(i -> new LocalPSWorker((long) i, updFunc, freq, epochs, batchSize, hyperParams, ec, ps))
				.collect(Collectors.toList());

		// Do data partition
		doDataPartition(ec, workers);

		// Create the worker threads
		workers.parallelStream().forEach(_ec::submit);

		// Wait for the worker finishing
		_ec.shutdown();
		try {
			_ec.awaitTermination(TIMEOUT, TimeUnit.MINUTES);
		} catch (InterruptedException e) {
			throw new DMLRuntimeException(
					String.format("ParamservBuiltinCPInstruction: an error occur: %s", e.getMessage()));
		}

		// Create the output
		ListObject result;
		try {
			result = ps.getResult();
		} catch (InterruptedException e) {
			throw new DMLRuntimeException(
					String.format("ParamservBuiltinCPInstruction: an error occur: %s", e.getMessage()));
		}
		ec.setVariable(output.getName(), result);
	}

	private PSUpdateType getUpdateType() {
		PSUpdateType updType = PSUpdateType.valueOf(getParam(PS_UPDATE_TYPE));
		switch (updType) {
		case ASP:
		case SSP:
			throw new DMLRuntimeException(String.format("Not support update type '%s'.", updType));
		case BSP:
			break;
		}
		return updType;
	}

	private PSFrequency getFrequency() {
		if (!getParameterMap().containsKey(PS_FREQUENCY)) {
			return DEFAULT_UPDATE_FREQUENCY;
		}
		PSFrequency freq = PSFrequency.valueOf(getParam(PS_FREQUENCY));
		switch (freq) {
		case EPOCH:
			throw new DMLRuntimeException("Not support epoch update frequency.");
		case BATCH:
			break;
		}
		return freq;
	}

	/**
	 * Get the worker numbers according to the vcores
	 *
	 * @param mode execution mode
	 * @return worker numbers
	 */
	private int getWorkerNum(PSModeType mode) {
		int workerNum = DEFAULT_LEVEL_PARALLELISM;
		if (getParameterMap().containsKey(PS_PARALLELISM)) {
			workerNum = Integer.valueOf(getParam(PS_PARALLELISM));
		}
		switch (mode) {
		case LOCAL:
			//FIXME: this is a workaround for a maximum number of buffers in openblas
			//However, the root cause is a missing function preparation for each worker
			//(i.e., deep copy with unique file names, and reduced degree of parallelism)
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
	private ParamServer createPS(PSModeType mode, String aggFunc, PSFrequency freq, PSUpdateType updateType,
			int workerNum, ListObject model, ExecutionContext ec, ListObject hyperParams) {
		ParamServer ps = null;
		switch (mode) {
		case LOCAL:
			ps = new LocalParamServer(model, aggFunc, freq, updateType, ec, workerNum, hyperParams);
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
		ListObject hyperparams = null;
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
		List<MatrixObject> pfs = disjointContiguous(workers.size(), features);
		List<MatrixObject> pls = disjointContiguous(workers.size(), labels);
		if (pfs.size() < workers.size()) {
			LOG.warn(String.format(
					"There is only %d batches of data but has %d workers. Hence, reset the number of workers with %d.",
					pfs.size(), workers.size(), pfs.size()));
			workers = workers.subList(0, pfs.size());
		}
		for (int i = 0; i < workers.size(); i++) {
			workers.get(i).setFeatures(pfs.get(i));
			workers.get(i).setLabels(pls.get(i));
		}

		// validation data
		List<MatrixObject> pvfs = disjointContiguous(workers.size(), valFeatures);
		List<MatrixObject> pvls = disjointContiguous(workers.size(), valLabels);
		for (int i = 0; i < workers.size(); i++) {
			workers.get(i).setValFeatures(pvfs.get(i));
			workers.get(i).setValLabels(pvls.get(i));
		}
	}

	private List<MatrixObject> disjointContiguous(int workerNum, MatrixObject mo) {
		List<MatrixObject> list = new ArrayList<>();
		long stepSize = (long) Math.ceil(mo.getNumRows() / workerNum);
		long begin = 1;
		while (begin < mo.getNumRows()) {
			long end = Math.min(begin + stepSize, mo.getNumRows());
			MatrixObject pmo = ParamservUtils.sliceMatrix(mo, begin, end);
			list.add(pmo);
			begin = end + 1;
		}
		return list;
	}
}
