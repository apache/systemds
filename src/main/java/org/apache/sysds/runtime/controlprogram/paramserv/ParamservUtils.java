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

package org.apache.sysds.runtime.controlprogram.paramserv;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.MultiThreadedHop;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.ForProgramBlock;
import org.apache.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysds.runtime.controlprogram.IfProgramBlock;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.POptMode;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.util.ProgramConverter;

public class ParamservUtils {

	protected static final Log LOG = LogFactory.getLog(ParamservUtils.class.getName());
	public static final String PS_FUNC_PREFIX = "_ps_";
	public static long SEED = -1; // Used for generating permutation

	/**
	 * Deep copy the list object
	 *
	 * @param lo list object
	 * @param cleanup clean up the given list object
	 * @return a new copied list object
	 */
	public static ListObject copyList(ListObject lo, boolean cleanup) {
		List<Data> newData = IntStream.range(0, lo.getLength()).mapToObj(i -> {
			Data oldData = lo.slice(i);
			if (oldData instanceof MatrixObject)
				return createShallowCopy((MatrixObject) oldData);
			else if (oldData instanceof ListObject || oldData instanceof FrameObject)
				throw new DMLRuntimeException("Copy list: does not support list or frame.");
			else
				return oldData;
		}).collect(Collectors.toList());
		ListObject result = new ListObject(newData, lo.getNames());
		if (cleanup)
			ParamservUtils.cleanupListObject(lo);
		return result;
	}

	/**
	 * Clean up the list object according to its own data status
	 * @param ec execution context
	 * @param lName list var name
	 */
	public static void cleanupListObject(ExecutionContext ec, String lName) {
		ListObject lo = (ListObject) ec.removeVariable(lName);
		cleanupListObject(ec, lo, lo.getStatus());
	}

	/**
	 * Clean up the list object according to the given array of data status (i.e., false {@literal =>} not be removed)
	 * @param ec execution context
	 * @param lName list var name
	 * @param status data status
	 */
	public static void cleanupListObject(ExecutionContext ec, String lName, boolean[] status) {
		ListObject lo = (ListObject) ec.removeVariable(lName);
		cleanupListObject(ec, lo, status);
	}

	public static void cleanupListObject(ExecutionContext ec, ListObject lo) {
		cleanupListObject(ec, lo, lo.getStatus());
	}

	public static void cleanupListObject(ExecutionContext ec, ListObject lo, boolean[] status) {
		for (int i = 0; i < lo.getLength(); i++) {
			if (status != null && !status[i])
				continue; // data ref by other object must not be cleaned up
			ParamservUtils.cleanupData(ec, lo.getData().get(i));
		}
	}

	public static void cleanupData(ExecutionContext ec, Data data) {
		if (!(data instanceof CacheableData))
			return;
		CacheableData<?> cd = (CacheableData<?>) data;
		cd.enableCleanup(true);
		ec.cleanupCacheableData(cd);
	}

	public static void cleanupData(ExecutionContext ec, String varName) {
		cleanupData(ec, ec.removeVariable(varName));
	}

	public static void cleanupListObject(ListObject lo) {
		cleanupListObject(ExecutionContextFactory.createContext(), lo);
	}

	public static MatrixObject newMatrixObject(MatrixBlock mb) {
		return newMatrixObject(mb, true);
	}
	
	public static MatrixObject newMatrixObject(MatrixBlock mb, boolean cleanup) {
		MatrixObject result = new MatrixObject(ValueType.FP64, OptimizerUtils.getUniqueTempFileName(),
			new MetaDataFormat(new MatrixCharacteristics(-1, -1, ConfigurationManager.getBlocksize(),
			ConfigurationManager.getBlocksize()), FileFormat.BINARY));
		result.acquireModify(mb);
		result.release();
		result.enableCleanup(cleanup);
		return result;
	}
	
	public static MatrixObject createShallowCopy(MatrixObject mo) {
		return newMatrixObject(mo.acquireReadAndRelease(), false);
	}

	/**
	 * Slice the matrix
	 *
	 * @param mo input matrix
	 * @param rl low boundary
	 * @param rh high boundary
	 * @return new sliced matrix
	 */
	public static MatrixObject sliceMatrix(MatrixObject mo, long rl, long rh) {
		MatrixBlock mb = mo.acquireReadAndRelease();
		return newMatrixObject(sliceMatrixBlock(mb, rl, rh), false);
	}

	/**
	 * Slice the matrix block and return a matrix block
	 * (used in spark)
	 *
	 * @param mb input matrix
	 * @param rl low boundary
	 * @param rh high boundary
	 * @return new sliced matrix block
	 */
	public static MatrixBlock sliceMatrixBlock(MatrixBlock mb, long rl, long rh) {
		return mb.slice((int) rl - 1, (int) rh - 1);
	}

	/**
	 * Generate the permutation
	 * @param numEntries permutation size
	 * @param seed seed used to generate random number
	 * @return permutation matrix
	 */
	public static MatrixBlock generatePermutation(int numEntries, long seed) {
		// Create a sequence and sample w/o replacement
		// (no need to materialize the sequence because ctable only uses its meta data)
		MatrixBlock seq = new MatrixBlock(numEntries, 1, false);
		MatrixBlock sample = MatrixBlock.sampleOperations(numEntries, numEntries, false, seed);

		// Combine the sequence and sample as a table
		return seq.ctableSeqOperations(sample, 1.0, new MatrixBlock(numEntries, numEntries, true));
	}

	/**
	 * Generates a matrix which when left multiplied with the input matrix will subsample
	 * @param nsamples number of samples
	 * @param nrows number of rows in input matrix
	 * @param seed seed used to generate random number
	 * @return subsample matrix
	 */
	public static MatrixBlock generateSubsampleMatrix(int nsamples, int nrows, long seed) {
		MatrixBlock seq = new MatrixBlock(nsamples, nrows, false);
		// No replacement to preserve as much of the original data as possible
		MatrixBlock sample = MatrixBlock.sampleOperations(nrows, nsamples, false, seed);
		return seq.ctableSeqOperations(sample, 1.0, new MatrixBlock(nsamples, nrows, true), false);
	}

	/**
	 * Generates a matrix which when left multiplied with the input matrix will replicate n data rows
	 * @param nsamples number of samples
	 * @param nrows number of rows in input matrix
	 * @param seed seed used to generate random number
	 * @return replication matrix
	 */
	public static MatrixBlock generateReplicationMatrix(int nsamples, int nrows, long seed) {
		MatrixBlock seq = new MatrixBlock(nsamples, nrows, false);
		// Replacement set to true to provide random replication
		MatrixBlock sample = MatrixBlock.sampleOperations(nrows, nsamples, true, seed);
		return seq.ctableSeqOperations(sample, 1.0, new MatrixBlock(nsamples, nrows, true), false);
	}

	public static ExecutionContext createExecutionContext(ExecutionContext ec,
		LocalVariableMap varsMap, String updFunc, String aggFunc, int k)
	{
		return createExecutionContext(ec, varsMap, updFunc, aggFunc, k, false);
	}

	public static ExecutionContext createExecutionContext(ExecutionContext ec,
		LocalVariableMap varsMap, String updFunc, String aggFunc, int k, boolean forceExecTypeCP)
	{
		Program prog = ec.getProgram();

		// 1. Recompile the internal program blocks 
		recompileProgramBlocks(k, prog.getProgramBlocks(), forceExecTypeCP);
		// 2. Recompile the imported function blocks
		boolean opt = prog.getFunctionProgramBlocks(false).isEmpty();
		prog.getFunctionProgramBlocks(opt)
			.forEach((fname, fvalue) -> recompileProgramBlocks(k, fvalue.getChildBlocks(), forceExecTypeCP));

		// 3. Copy all functions 
		return ExecutionContextFactory.createContext(
			new LocalVariableMap(varsMap), copyProgramFunctions(prog));
	}

	public static List<ExecutionContext> copyExecutionContext(ExecutionContext ec, int num) {
		return IntStream.range(0, num).mapToObj(i ->
			ExecutionContextFactory.createContext(
				new LocalVariableMap(ec.getVariables()),
				copyProgramFunctions(ec.getProgram()))
		).collect(Collectors.toList());
	}
	
	private static Program copyProgramFunctions(Program prog) {
		Program newProg = new Program(prog.getDMLProg());
		boolean opt = prog.getFunctionProgramBlocks(false).isEmpty();
		for( Entry<String, FunctionProgramBlock> e : prog.getFunctionProgramBlocks(opt).entrySet() ) {
			String[] parts = DMLProgram.splitFunctionKey(e.getKey());
			FunctionProgramBlock fpb = ProgramConverter
				.createDeepCopyFunctionProgramBlock(e.getValue(), new HashSet<>(), new HashSet<>());
			fpb._namespace = parts[0];
			fpb._functionName = parts[1];
			newProg.addFunctionProgramBlock(parts[0], parts[1], fpb, opt);
			newProg.addProgramBlock(fpb);
		}
		return newProg;
	}

	public static void recompileProgramBlocks(int k, List<ProgramBlock> pbs) {
		recompileProgramBlocks(k, pbs, false);
	}

	public static void recompileProgramBlocks(int k, List<ProgramBlock> pbs, boolean forceExecTypeCP) {
		// Reset the visit status from root
		for (ProgramBlock pb : pbs)
			DMLTranslator.resetHopsDAGVisitStatus(pb.getStatementBlock());

		// Should recursively assign the level of parallelism
		// and recompile the program block
		try {
			if(forceExecTypeCP)
				rAssignParallelismAndRecompile(pbs, k, true, forceExecTypeCP);
			else
				rAssignParallelismAndRecompile(pbs, k, false, forceExecTypeCP);
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}

	private static boolean rAssignParallelismAndRecompile(List<ProgramBlock> pbs, int k, boolean recompiled, boolean forceExecTypeCP) throws IOException {
		for (ProgramBlock pb : pbs) {
			if (pb instanceof ParForProgramBlock) {
				ParForProgramBlock pfpb = (ParForProgramBlock) pb;
				if( !pfpb.isDegreeOfParallelismFixed() ) {
					pfpb.setDegreeOfParallelism(k);
					if( k == 1 )
						pfpb.setOptimizationMode(POptMode.NONE);
					recompiled |= rAssignParallelismAndRecompile(pfpb.getChildBlocks(), 1, recompiled, forceExecTypeCP);
				}
			} else if (pb instanceof ForProgramBlock) {
				recompiled |= rAssignParallelismAndRecompile(((ForProgramBlock) pb).getChildBlocks(), k, recompiled, forceExecTypeCP);
			} else if (pb instanceof WhileProgramBlock) {
				recompiled |= rAssignParallelismAndRecompile(((WhileProgramBlock) pb).getChildBlocks(), k, recompiled, forceExecTypeCP);
			} else if (pb instanceof FunctionProgramBlock) {
				recompiled |= rAssignParallelismAndRecompile(((FunctionProgramBlock) pb).getChildBlocks(), k, recompiled, forceExecTypeCP);
			} else if (pb instanceof IfProgramBlock) {
				IfProgramBlock ipb = (IfProgramBlock) pb;
				recompiled |= rAssignParallelismAndRecompile(ipb.getChildBlocksIfBody(), k, recompiled, forceExecTypeCP);
				if (ipb.getChildBlocksElseBody() != null)
					recompiled |= rAssignParallelismAndRecompile(ipb.getChildBlocksElseBody(), k, recompiled, forceExecTypeCP);
			} else {
				StatementBlock sb = pb.getStatementBlock();
				for (Hop hop : sb.getHops())
					recompiled |= rAssignParallelismAndRecompile(hop, k, recompiled);
			}
			// Recompile the program block
			if (recompiled) {
				if(forceExecTypeCP)
					Recompiler.rRecompileProgramBlock2Forced(pb, pb.getThreadID(), new HashSet<>(), ExecType.CP);
				else
					Recompiler.recompileProgramBlockInstructions(pb);
			}
		}
		return recompiled;
	}

	private static boolean rAssignParallelismAndRecompile(Hop hop, int k, boolean recompiled) {
		if (hop.isVisited()) {
			return recompiled;
		}
		if (hop instanceof MultiThreadedHop) {
			// Reassign the level of parallelism
			MultiThreadedHop mhop = (MultiThreadedHop) hop;
			mhop.setMaxNumThreads(k);
			recompiled = true;
		}
		ArrayList<Hop> inputs = hop.getInput();
		for (Hop h : inputs) {
			recompiled |= rAssignParallelismAndRecompile(h, k, recompiled);
		}
		hop.setVisited();
		return recompiled;
	}

	@SuppressWarnings("unused")
	private static FunctionProgramBlock getFunctionBlock(ExecutionContext ec, String funcName) {
		String[] cfn = DMLProgram.splitFunctionKey(funcName);
		String ns = cfn[0];
		String fname = cfn[1];
		return ec.getProgram().getFunctionProgramBlock(ns, fname);
	}

	public static MatrixBlock cbindMatrix(MatrixBlock left, MatrixBlock right) {
		return left.append(right, new MatrixBlock());
	}


	/**
	 * Accumulate the given gradients into the accrued gradients
	 *
	 * @param accGradients accrued gradients list object
	 * @param gradients given gradients list object
	 * @param cleanup clean up the given gradients list object
	 * @return new accrued gradients list object
	 */
	public static ListObject accrueGradients(ListObject accGradients, ListObject gradients, boolean cleanup) {
		return accrueGradients(accGradients, gradients, false, cleanup);
	}

	/**
	 * Accumulate the given gradients into the accrued gradients
	 *
	 * @param accGradients accrued gradients list object
	 * @param gradients given gradients list object
	 * @param par parallel execution
	 * @param cleanup clean up the given gradients list object
	 * @return new accrued gradients list object
	 */
	public static ListObject accrueGradients(ListObject accGradients, ListObject gradients, boolean par, boolean cleanup) {
		if (accGradients == null)
			return ParamservUtils.copyList(gradients, cleanup);
		IntStream range = IntStream.range(0, accGradients.getLength());
		(par ? range.parallel() : range).forEach(i -> {
			MatrixBlock mb1 = ((MatrixObject) accGradients.getData().get(i)).acquireReadAndRelease();
			MatrixBlock mb2 = ((MatrixObject) gradients.getData().get(i)).acquireReadAndRelease();
			mb1.binaryOperationsInPlace(new BinaryOperator(Plus.getPlusFnObject()), mb2);
		});
		if (cleanup)
			ParamservUtils.cleanupListObject(gradients);
		return accGradients;
	}

	/**
	 * Accumulate the given models into the accrued accrueModels
	 *
	 * @param accModels accrued models list object
	 * @param model given models list object
	 * @param cleanup clean up the given models list object
	 * @return new accrued models list object
	 */
	public static ListObject accrueModels(ListObject accModels, ListObject model, boolean cleanup) {
		return accrueModels(accModels, model, false, cleanup);
	}

	/**
	 * Accumulate the given models into the accrued models
	 *
	 * @param accModels accrued models list object
	 * @param model given models list object
	 * @param par parallel execution
	 * @param cleanup clean up the given models list object
	 * @return new accrued models list object
	 */
	public static ListObject accrueModels(ListObject accModels, ListObject model, boolean par, boolean cleanup) {
		if (accModels == null)
			return ParamservUtils.copyList(model, cleanup);
		IntStream range = IntStream.range(0, accModels.getLength());
		(par ? range.parallel() : range).forEach(i -> {
			MatrixBlock mb1 = ((MatrixObject) accModels.getData().get(i)).acquireReadAndRelease();
			MatrixBlock mb2 = ((MatrixObject) model.getData().get(i)).acquireReadAndRelease();
			mb1.binaryOperationsInPlace(new BinaryOperator(Plus.getPlusFnObject()), mb2);
		});
		if (cleanup)
			ParamservUtils.cleanupListObject(model);
		return accModels;
	}
}
