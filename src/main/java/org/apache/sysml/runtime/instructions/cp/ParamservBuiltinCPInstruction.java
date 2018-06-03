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
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.controlprogram.paramserv.LocalPSWorker;
import org.apache.sysml.runtime.controlprogram.paramserv.LocalParamServer;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamServer;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.matrix.operators.Operator;

public class ParamservBuiltinCPInstruction extends ParameterizedBuiltinCPInstruction {

	private static final int DEFAULT_BATCH_SIZE = 64;
	private static final PSFrequency DEFAULT_UPDATE_FREQUENCY = PSFrequency.BATCH;
	private static final PSScheme DEFAULT_SCHEME = PSScheme.DISJOINT_CONTIGUOUS;

	//internal local debug level
	private static final boolean LDEBUG = false;

	private ExecutorService _es;
	private ParamServer _ps;
	public static final int TIMEOUT = Integer.MAX_VALUE;

	static {
		// for internal debugging only
		if (LDEBUG) {
			Logger.getLogger("org.apache.sysml.runtime.controlprogram.paramserv").setLevel(Level.DEBUG);
		}
	}

	/**
	 * A thread error handler for workers and agg service
	 */
	public class PSErrorHandler implements Function<Throwable, Void> {

		private List<Throwable> error = new ArrayList<>();

		boolean hasError() {
			return !error.isEmpty();
		}

		String getError() {
			StringBuilder sb = new StringBuilder();
			error.forEach(e -> sb.append(ExceptionUtils.getFullStackTrace(e)).append("\n"));
			return sb.toString();
		}

		@Override
		public Void apply(Throwable throwable) {
			if (throwable != null) {
				error.add(throwable);
				// shutdown all the workers and agg service
				_es.shutdownNow();
				_ps.shutdown();
			}
			return null;
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
		_es = Executors.newFixedThreadPool(workerNum);
		String updFunc = getParam(PS_UPDATE_FUN);
		String aggFunc = getParam(PS_AGGREGATION_FUN);

		// Create the worker execution context
		int k = getRemainingCores() / workerNum;
		List<ExecutionContext> workerECs = createExecutionContext(ec, updFunc, workerNum, k);
		// Create the agg service execution context
		ExecutionContext aggServiceEC = createExecutionContext(ec, aggFunc, 1, 1).get(0);

		PSFrequency freq = getFrequency();
		PSUpdateType updateType = getUpdateType();
		int epochs = Integer.valueOf(getParam(PS_EPOCHS));
		if (epochs <= 0) {
			throw new DMLRuntimeException(
					String.format("Paramserv function: The argument '%s' could not be less than or equal to 0.",
							PS_EPOCHS));
		}

		// Create an error handler
		PSErrorHandler handler = new PSErrorHandler();

		// Create the parameter server
		ListObject model = ec.getListObject(getParam(PS_MODEL));
		_ps = createPS(mode, aggFunc, freq, updateType, workerNum, model, aggServiceEC, handler);

		// Create the local workers
		List<LocalPSWorker> workers = IntStream.range(0, workerNum).mapToObj(
				i -> new LocalPSWorker((long) i, updFunc, freq, epochs, getBatchSize(), workerECs.get(i), _ps))
				.collect(Collectors.toList());

		// Do data partition
		doDataPartition(ec, workers);

		// Create the worker threads
		workers.forEach(w -> CompletableFuture.runAsync(w, _es).exceptionally(handler));

		// Wait for the worker finishing
		_es.shutdown();
		try {
			_es.awaitTermination(TIMEOUT, TimeUnit.DAYS);
		} catch (InterruptedException e) {
			throw new DMLRuntimeException(
					String.format("ParamservBuiltinCPInstruction: an error occurred: %s", e.getMessage()));
		}

		// If failed
		if (handler.hasError()) {
			throw new DMLRuntimeException(
					String.format("ParamservBuiltinCPInstruction: some error occurred: %s", handler.getError()));
		}

		// Create the output
		ListObject result;
		try {
			result = _ps.getResult();
		} catch (InterruptedException e) {
			throw new DMLRuntimeException(
					String.format("ParamservBuiltinCPInstruction: an error occurred: %s", e.getMessage()));
		}
		ec.setVariable(output.getName(), result);
	}

	private List<ExecutionContext> createExecutionContext(ExecutionContext ec, String funcName, int workerNum, int k) {
		// Fetch the target function
		String[] keys = DMLProgram.splitFunctionKey(funcName);
		FunctionProgramBlock targetFunc = ec.getProgram().getFunctionProgramBlock(keys[0], keys[1]);
		ProgramBlock pb = targetFunc.getChildBlocks().get(0);

		// BFS travel all the hops
		LinkedList<Hop> hops = new LinkedList<>(pb.getStatementBlock().getHops());
		while (!hops.isEmpty()) {
			Hop hop = hops.remove(0);
			if (hop instanceof Hop.MultiThreadedHop) {
				// Reassign the level of parallelism
				Hop.MultiThreadedHop mhop = (Hop.MultiThreadedHop) hop;
				mhop.setMaxNumThreads(k);
			}
			hops.addAll(hop.getInput());
		}

		// Create a new program,
		// and only put the target function
		Program prog = new Program();
		FunctionProgramBlock copiedPB = new FunctionProgramBlock(prog, targetFunc.getInputParams(),
				targetFunc.getOutputParams());
		prog.addProgramBlock(copiedPB);
		prog.addFunctionProgramBlock(keys[0], keys[1], copiedPB);
		copiedPB.setChildBlocks(new ArrayList<>(Collections.singletonList(pb)));

		return IntStream.range(0, workerNum).mapToObj(i -> {
			// Put the hyperparam into the variables table
			LocalVariableMap varsMap = new LocalVariableMap();
			ListObject hyperParams = getHyperParams(ec);
			if (hyperParams != null) {
				varsMap.put(PS_HYPER_PARAMS, hyperParams);
			}
			ExecutionContext newEC = ExecutionContextFactory.createContext(varsMap, prog);
			// Recompile the program block
			Recompiler.recompileHopsDag(copiedPB.getChildBlocks().get(0).getStatementBlock(),
					copiedPB.getChildBlocks().get(0).getStatementBlock().getHops(), newEC.getVariables(), null, false,
					false, 0);
			return newEC;
		}).collect(Collectors.toList());
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

	private int getRemainingCores() {
		return InfrastructureAnalyzer.getLocalParallelism() - 1;
	}

	/**
	 * Get the worker numbers according to the vcores
	 *
	 * @param mode execution mode
	 * @return worker numbers
	 */
	private int getWorkerNum(PSModeType mode) {
		int workerNum = -1;
		switch (mode) {
		case LOCAL:
			// default worker number: available cores - 1 (assign one process for agg service)
			workerNum = getRemainingCores();
			if (getParameterMap().containsKey(PS_PARALLELISM)) {
				workerNum = Math.min(workerNum, Integer.valueOf(getParam(PS_PARALLELISM)));
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
			int workerNum, ListObject model, ExecutionContext ec, PSErrorHandler handler) {
		ParamServer ps = null;
		switch (mode) {
		case LOCAL:
			ps = new LocalParamServer(model, aggFunc, freq, updateType, ec, workerNum, handler);
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
