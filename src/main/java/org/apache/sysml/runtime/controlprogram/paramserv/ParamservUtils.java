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

package org.apache.sysml.runtime.controlprogram.paramserv;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.ForProgramBlock;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.IfProgramBlock;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;
import org.apache.sysml.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.caching.FrameObject;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.controlprogram.parfor.ProgramConverter;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.ListObject;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MetaDataFormat;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;

public class ParamservUtils {

	public static final String UPDATE_FUNC_PREFIX = "_worker_";
	public static final String AGG_FUNC_PREFIX = "_agg_";

	/**
	 * Deep copy the list object
	 *
	 * @param lo list object
	 * @return a new copied list object
	 */
	public static ListObject copyList(ListObject lo) {
		if (lo.getLength() == 0) {
			return lo;
		}
		List<Data> newData = IntStream.range(0, lo.getLength()).mapToObj(i -> {
			Data oldData = lo.slice(i);
			if (oldData instanceof MatrixObject) {
				MatrixObject mo = (MatrixObject) oldData;
				return sliceMatrix(mo, 1, mo.getNumRows());
			} else if (oldData instanceof ListObject || oldData instanceof FrameObject) {
				throw new DMLRuntimeException("Copy list: does not support list or frame.");
			} else {
				return oldData;
			}
		}).collect(Collectors.toList());
		return new ListObject(newData, lo.getNames());
	}

	public static void cleanupListObject(ExecutionContext ec, String lName) {
		ListObject lo = (ListObject) ec.removeVariable(lName);
		lo.getData().forEach(ParamservUtils::cleanupData);
	}

	public static void cleanupData(Data data) {
		if (!(data instanceof CacheableData))
			return;
		CacheableData<?> cd = (CacheableData<?>) data;
		cd.enableCleanup(true);
		cd.clearData();
	}

	public static MatrixObject newMatrixObject() {
		return new MatrixObject(Expression.ValueType.DOUBLE, OptimizerUtils.getUniqueTempFileName(), new MetaDataFormat(new MatrixCharacteristics(-1, -1, -1, -1), OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
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
		MatrixObject result = newMatrixObject();
		MatrixBlock tmp = mo.acquireRead();
		result.acquireModify(tmp.slice((int) rl - 1, (int) rh - 1));
		mo.release();
		result.release();
		result.enableCleanup(false);
		return result;
	}

	public static MatrixBlock generatePermutation(int numEntries) {
		// Create a sequence and sample w/o replacement
		MatrixBlock seq = MatrixBlock.seqOperations(1, numEntries, 1);
		MatrixBlock sample = MatrixBlock.sampleOperations(numEntries, numEntries, false, -1);

		// Combine the sequence and sample as a table
		MatrixBlock permutation = new MatrixBlock(numEntries, numEntries, true);
		seq.ctableOperations(null, sample, 1.0, permutation);
		return permutation;
	}

	public static ExecutionContext createExecutionContext(ExecutionContext ec, String updFunc, String aggFunc, int workerNum, int k) {
		FunctionProgramBlock updPB = getFunctionBlock(ec, updFunc);
		FunctionProgramBlock aggPB = getFunctionBlock(ec, aggFunc);

		Program prog = ec.getProgram();

		// 1. Recompile the internal program blocks
		recompileProgramBlocks(k, prog.getProgramBlocks());
		// 2. Recompile the imported function blocks
		prog.getFunctionProgramBlocks().forEach((fname, fvalue) -> recompileProgramBlocks(k, fvalue.getChildBlocks()));

		// Copy function for workers
		IntStream.range(0, workerNum).forEach(i -> {
			copyFunction(updFunc, updPB, prog, UPDATE_FUNC_PREFIX + i + "_");
		});

		// Copy function for agg service
		copyFunction(aggFunc, aggPB, prog, AGG_FUNC_PREFIX);

		return ExecutionContextFactory.createContext(prog);
	}

	private static void copyFunction(String funcName, FunctionProgramBlock updPB, Program prog, String prefix) {
		String[] keys = DMLProgram.splitFunctionKey(funcName);
		String namespace = null;
		String func = keys[0];
		if (keys.length == 2) {
			namespace = keys[0];
			func = keys[1];
		}
		FunctionProgramBlock copiedFunc = ProgramConverter
				.createDeepCopyFunctionProgramBlock(updPB, new HashSet<>(), new HashSet<>());
		String fnameNew = prefix + func;
		prog.addFunctionProgramBlock(namespace, fnameNew, copiedFunc);
	}

	private static void recompileProgramBlocks(int k, ArrayList<ProgramBlock> pbs) {
		// Reset the visit status from root
		for (ProgramBlock pb : pbs)
			DMLTranslator.resetHopsDAGVisitStatus(pb.getStatementBlock());

		// Should recursively assign the level of parallelism
		// and recompile the program block
		try {
			rAssignParallelism(pbs, k, false);
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}

	private static boolean rAssignParallelism(ArrayList<ProgramBlock> pbs, int k, boolean recompiled) throws IOException {
		for (ProgramBlock pb : pbs) {
			if (pb instanceof ParForProgramBlock) {
				ParForProgramBlock pfpb = (ParForProgramBlock) pb;
				pfpb.setDegreeOfParallelism(k);
				recompiled |= rAssignParallelism(pfpb.getChildBlocks(), 1, recompiled);
			} else if (pb instanceof ForProgramBlock) {
				recompiled |= rAssignParallelism(((ForProgramBlock) pb).getChildBlocks(), k, recompiled);
			} else if (pb instanceof WhileProgramBlock) {
				recompiled |= rAssignParallelism(((WhileProgramBlock) pb).getChildBlocks(), k, recompiled);
			} else if (pb instanceof FunctionProgramBlock) {
				recompiled |= rAssignParallelism(((FunctionProgramBlock) pb).getChildBlocks(), k, recompiled);
			} else if (pb instanceof IfProgramBlock) {
				IfProgramBlock ipb = (IfProgramBlock) pb;
				recompiled |= rAssignParallelism(ipb.getChildBlocksIfBody(), k, recompiled);
				if (ipb.getChildBlocksElseBody() != null)
					recompiled |= rAssignParallelism(ipb.getChildBlocksElseBody(), k, recompiled);
			} else {
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

	private static boolean rAssignParallelism(Hop hop, int k, boolean recompiled) {
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


	private static FunctionProgramBlock getFunctionBlock(ExecutionContext ec, String funcName) {
		String[] keys = DMLProgram.splitFunctionKey(funcName);
		String namespace = null;
		String func = keys[0];
		if (keys.length == 2) {
			namespace = keys[0];
			func = keys[1];
		}
		return ec.getProgram().getFunctionProgramBlock(namespace, func);
	}
}
