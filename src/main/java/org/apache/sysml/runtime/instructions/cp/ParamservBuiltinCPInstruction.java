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

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.ForProgramBlock;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.IfProgramBlock;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;
import org.apache.sysml.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.controlprogram.paramserv.LocalPSWorker;
import org.apache.sysml.runtime.controlprogram.paramserv.LocalParamServer;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamServer;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysml.runtime.controlprogram.parfor.ProgramConverter;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.matrix.operators.Operator;

public class ParamservBuiltinCPInstruction extends ParameterizedBuiltinCPInstruction {

	private static final int DEFAULT_BATCH_SIZE = 64;
	private static final PSFrequency DEFAULT_UPDATE_FREQUENCY = PSFrequency.BATCH;
	private static final PSScheme DEFAULT_SCHEME = PSScheme.DISJOINT_CONTIGUOUS;

	//internal local debug level
	private static final boolean LDEBUG = false;
	protected static final Log LOG = LogFactory.getLog(ParamservBuiltinCPInstruction.class.getName());


	static {
		// for internal debugging only
		if (LDEBUG) {
			Logger.getLogger("org.apache.sysml.runtime.controlprogram.paramserv").setLevel(Level.DEBUG);
			Logger.getLogger(ParamservBuiltinCPInstruction.class.getName()).setLevel(Level.DEBUG);
		}
	}

	public ParamservBuiltinCPInstruction(Operator op, LinkedHashMap<String, String> paramsMap, CPOperand out, String opcode, String istr) {
		super(op, paramsMap, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		PSModeType mode = getPSMode();
		int workerNum = getWorkerNum(mode);
		ExecutorService es = Executors.newFixedThreadPool(workerNum);
		String updFunc = getParam(PS_UPDATE_FUN);
		String aggFunc = getParam(PS_AGGREGATION_FUN);

		// Create the workers' execution context
		int k = getParLevel(workerNum);
		List<ExecutionContext> workerECs = createExecutionContext(ec, updFunc, workerNum, k);

		// Create the agg service's execution context
		ExecutionContext aggServiceEC = createExecutionContext(ec, aggFunc, 1, 1).get(0);

		PSFrequency freq = getFrequency();
		PSUpdateType updateType = getUpdateType();
		int epochs = getEpochs();

		// Create the parameter server
		ListObject model = ec.getListObject(getParam(PS_MODEL));
		ParamServer ps = createPS(mode, aggFunc, updateType, workerNum, model, aggServiceEC);

		// Create the local workers
		List<LocalPSWorker> workers = IntStream.range(0, workerNum)
			.mapToObj(i -> new LocalPSWorker(i, updFunc, freq, epochs, getBatchSize(), workerECs.get(i), ps))
			.collect(Collectors.toList());

		// Do data partition
		doDataPartition(ec, workers);

		if (LOG.isDebugEnabled()) {
			LOG.debug(String.format("\nConfiguration of paramserv func: \nmode: %s \nworkerNum: %d \nupdate frequency: %s \nstrategy: %s",
					mode, workerNum, freq, updateType));
		}

		// Launch the worker threads and wait for completion
		try {
			for (Future<Void> ret : es.invokeAll(workers))
				ret.get(); //error handling
		} catch (InterruptedException | ExecutionException e) {
			throw new DMLRuntimeException("ParamservBuiltinCPInstruction: some error occurred: ", e);
		} finally {
			es.shutdownNow();
		}

		// Fetch the final model from ps
		ListObject result;
		result = ps.getResult();
		ec.setVariable(output.getName(), result);
	}

	private PSModeType getPSMode() {
		PSModeType mode;
		try {
			mode = PSModeType.valueOf(getParam(PS_MODE));
		} catch (IllegalArgumentException e) {
			throw new DMLRuntimeException(String.format("Paramserv function: not support ps execution mode '%s'", getParam(PS_MODE)));
		}
		if( mode == PSModeType.REMOTE_SPARK )
			throw new DMLRuntimeException("Do not support remote spark.");
		return mode;
	}

	private int getEpochs() {
		int epochs = Integer.valueOf(getParam(PS_EPOCHS));
		if (epochs <= 0) {
			throw new DMLRuntimeException(String.format("Paramserv function: "
				+ "The argument '%s' could not be less than or equal to 0.", PS_EPOCHS));
		}
		return epochs;
	}

	private int getParLevel(int workerNum) {
		return Math.max((int)Math.ceil((double)getRemainingCores()/workerNum), 1);
	}

	private List<ExecutionContext> createExecutionContext(ExecutionContext ec, String funcName, int workerNum, int k) {
		// Fetch the target function
		String[] keys = DMLProgram.splitFunctionKey(funcName);
		String namespace = null;
		String func = keys[0];
		if (keys.length == 2) {
			namespace = keys[0];
			func = keys[1];
		}
		return createExecutionContext(ec, namespace, func, workerNum, k);
	}

	private List<ExecutionContext> createExecutionContext(ExecutionContext ec, String namespace, String func,
			int workerNum, int k) {
		FunctionProgramBlock targetFunc = ec.getProgram().getFunctionProgramBlock(namespace, func);
		return IntStream.range(0, workerNum).mapToObj(i -> {
			// Put the hyperparam into the variables table
			LocalVariableMap varsMap = new LocalVariableMap();
			ListObject hyperParams = getHyperParams(ec);
			if (hyperParams != null) {
				varsMap.put(PS_HYPER_PARAMS, hyperParams);
			}

			// Deep copy the target func
			FunctionProgramBlock copiedFunc = ProgramConverter
				.createDeepCopyFunctionProgramBlock(targetFunc, new HashSet<>(), new HashSet<>());

			// Reset the visit status from root
			for( ProgramBlock pb : copiedFunc.getChildBlocks() )
				DMLTranslator.resetHopsDAGVisitStatus(pb.getStatementBlock());

			// Should recursively assign the level of parallelism
			// and recompile the program block
			try {
				rAssignParallelism(copiedFunc.getChildBlocks(), k, false);
			} catch (IOException e) {
				throw new DMLRuntimeException(e);
			}

			Program prog = new Program();
			prog.addProgramBlock(copiedFunc);
			prog.addFunctionProgramBlock(namespace, func, copiedFunc);
			return ExecutionContextFactory.createContext(varsMap, prog);

		}).collect(Collectors.toList());
	}

	private boolean rAssignParallelism(ArrayList<ProgramBlock> pbs, int k, boolean recompiled) throws IOException {
		for (ProgramBlock pb : pbs) {
			if (pb instanceof ParForProgramBlock) {
				ParForProgramBlock pfpb = (ParForProgramBlock) pb;
				pfpb.setDegreeOfParallelism(k);
				recompiled |= rAssignParallelism(pfpb.getChildBlocks(), 1, recompiled);
			}
			else if (pb instanceof ForProgramBlock) {
				recompiled |= rAssignParallelism(((ForProgramBlock) pb).getChildBlocks(), k, recompiled);
			}
			else if (pb instanceof WhileProgramBlock) {
				recompiled |= rAssignParallelism(((WhileProgramBlock) pb).getChildBlocks(), k, recompiled);
			}
			else if (pb instanceof FunctionProgramBlock) {
				recompiled |= rAssignParallelism(((FunctionProgramBlock) pb).getChildBlocks(), k, recompiled);
			}
			else if (pb instanceof IfProgramBlock) {
				IfProgramBlock ipb = (IfProgramBlock) pb;
				recompiled |= rAssignParallelism(ipb.getChildBlocksIfBody(), k, recompiled);
				if (ipb.getChildBlocksElseBody() != null)
					recompiled |= rAssignParallelism(ipb.getChildBlocksElseBody(), k, recompiled);
			}
			else {
				StatementBlock sb = pb.getStatementBlock();
				for (Hop hop : sb.getHops())
					recompiled |= rAssignParallelism(hop, k, recompiled);
			}
			// Recompile the program block
			if (recompiled) {
				Recompiler.recompileProgramBlockInstructions(pb);
			}
		}
		return recompiled;
	}

	private boolean rAssignParallelism(Hop hop, int k, boolean recompiled) {
		if (hop.isVisited()) {
			return recompiled;
		}
		if (hop instanceof Hop.MultiThreadedHop) {
			// Reassign the level of parallelism
			Hop.MultiThreadedHop mhop = (Hop.MultiThreadedHop) hop;
			mhop.setMaxNumThreads(k);
			recompiled = true;
		}
		ArrayList<Hop> inputs = hop.getInput();
		for (Hop h : inputs) {
			recompiled |= rAssignParallelism(h, k, recompiled);
		}
		hop.setVisited();
		return recompiled;
	}

	private PSUpdateType getUpdateType() {
		PSUpdateType updType;
		try {
			updType = PSUpdateType.valueOf(getParam(PS_UPDATE_TYPE));
		} catch (IllegalArgumentException e) {
			throw new DMLRuntimeException(String.format("Paramserv function: not support update type '%s'.", getParam(PS_UPDATE_TYPE)));
		}
		if (updType == PSUpdateType.SSP)
			throw new DMLRuntimeException("Not support update type SSP.");
		return updType;
	}

	private PSFrequency getFrequency() {
		if (!getParameterMap().containsKey(PS_FREQUENCY)) {
			return DEFAULT_UPDATE_FREQUENCY;
		}
		PSFrequency freq;
		try {
			freq = PSFrequency.valueOf(getParam(PS_FREQUENCY));
		} catch (IllegalArgumentException e) {
			throw new DMLRuntimeException(String.format("Paramserv function: not support '%s' update frequency.", getParam(PS_FREQUENCY)));
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
	private ParamServer createPS(PSModeType mode, String aggFunc, PSUpdateType updateType, int workerNum, ListObject model, ExecutionContext ec) {
		ParamServer ps = null;
		switch (mode) {
			case LOCAL:
				ps = new LocalParamServer(model, aggFunc, updateType, ec, workerNum);
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
			throw new DMLRuntimeException(String.format("Paramserv function: the number "
				+ "of argument '%s' could not be less than or equal to 0.", PS_BATCH_SIZE));
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
			try {
				scheme = PSScheme.valueOf(getParam(PS_SCHEME));
			} catch (IllegalArgumentException e) {
				throw new DMLRuntimeException(String.format("Paramserv function: not support data partition scheme '%s'", getParam(PS_SCHEME)));
			}
		}
		switch (scheme) {
		case DISJOINT_CONTIGUOUS:
			disjointContiguous(features, labels, valFeatures, valLabels, workers);
			break;
		case DISJOINT_RANDOM:
		case OVERLAP_RESHUFFLE:
		case DISJOINT_ROUND_ROBIN:
			throw new DMLRuntimeException(String.format("Paramserv function: the scheme '%s' is not supported.", scheme));
		}
	}

	private void disjointContiguous(MatrixObject features, MatrixObject labels, MatrixObject valFeatures,
			MatrixObject valLabels, List<LocalPSWorker> workers) {
		// training data
		List<MatrixObject> pfs = disjointContiguous(workers.size(), features);
		List<MatrixObject> pls = disjointContiguous(workers.size(), labels);
		if (pfs.size() < workers.size()) {
			if (LOG.isWarnEnabled()) {
				LOG.warn(String.format("There is only %d batches of data but has %d workers. "
					+ "Hence, reset the number of workers with %d.", pfs.size(), workers.size(), pfs.size()));
			}
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
