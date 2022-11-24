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

package org.apache.sysds.runtime.transform.tokenize;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.transform.tokenize.applier.TokenizerApplier;
import org.apache.sysds.runtime.transform.tokenize.applier.TokenizerApplierCount;
import org.apache.sysds.runtime.transform.tokenize.applier.TokenizerApplierHash;
import org.apache.sysds.runtime.transform.tokenize.applier.TokenizerApplierPosition;
import org.apache.sysds.runtime.transform.tokenize.builder.TokenizerBuilder;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.runtime.util.DependencyThreadPool;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;

public class Tokenizer implements Serializable {

	private static final long serialVersionUID = 7155673772374114577L;
	protected static final Log LOG = LogFactory.getLog(Tokenizer.class.getName());
	private static final boolean MULTI_THREADED_STAGES_TOKENIZER = false;
	public static final int TOKENIZE_NUM_BLOCKS = ConfigurationManager.getNumberTokenizeBlocks();

	private DocumentRepresentation[] internalRepresentation = null;
	private final TokenizerBuilder tokenizerBuilder;
	private final TokenizerApplier tokenizerApplier;

	protected Tokenizer(TokenizerBuilder tokenizerBuilder, TokenizerApplier tokenizerApplier) {
		this.tokenizerBuilder = tokenizerBuilder;
		this.tokenizerApplier = tokenizerApplier;
	}

	public Types.ValueType[] getSchema() {
		return tokenizerApplier.getOutSchema();
	}

	public int getMaxNumRows(int inRows) {
		return tokenizerApplier.getMaxNumRows(inRows);
	}

	public int getNumRowsEstimate(){
		// Estimate upperbound because e.g. Count Applier has less since it only outputs each unique token once
		if(internalRepresentation != null){
			if(tokenizerApplier.isWideFormat()) {
				return internalRepresentation.length;
			}
			if(tokenizerApplier.hasPadding()) {
				return internalRepresentation.length * tokenizerApplier.getMaxTokens();
			}
			return Arrays.stream(internalRepresentation).mapToInt(doc -> Math.min(doc.tokens.size(), tokenizerApplier.getMaxTokens())).sum();
		}
		throw new DMLRuntimeException("Internal Token Representation was not computed yet. Can not get exact size.");
	}

	public long getNumCols() {
		return tokenizerApplier.getNumCols();
	}

	public void allocateInternalRepresentation(int numDocuments){
		internalRepresentation = new DocumentRepresentation[numDocuments];
		tokenizerApplier.allocateInternalMeta(numDocuments);
	}

	public FrameBlock tokenize(FrameBlock in) {
		return tokenize(in, 1);
	}

	public FrameBlock tokenize(FrameBlock in, int k) {
		allocateInternalRepresentation(in.getNumRows());
		FrameBlock out = new FrameBlock(this.getSchema());
		if (k > 1 && !MULTI_THREADED_STAGES_TOKENIZER) {
			DependencyThreadPool pool = new DependencyThreadPool(k);
			LOG.debug("Tokenizing with full DAG on " + k + " Threads");
			try {
				List<DependencyTask<?>> tokenizeTasks = getTokenizeTasks(in, out, pool);
				int lastRow = pool.submitAllAndWait(tokenizeTasks).stream()//
					.map(s -> s == null? 0 :(Integer)s).max((x,y) -> Integer.compare(x, y)).get();
				if(lastRow != out.getNumRows()){
					out = out.slice(0, lastRow - 1, 0, out.getNumColumns() - 1, null);
				}
			} catch (ExecutionException | InterruptedException e) {
				LOG.error("MT tokenize failed");
				e.printStackTrace();
			}
			pool.shutdown();
		} else {
			build(in, k);
			out.ensureAllocatedColumns(tokenizerApplier.getNumRows(this.internalRepresentation));
			out = apply(out, k);
		}
		return out;
	}

	private List<DependencyTask<?>> getTokenizeTasks(FrameBlock in, FrameBlock out, DependencyThreadPool pool) {
		// TODO further optimisation of task graph to reduce memory usage!
		// TODO add cache awareness
		List<DependencyTask<?>> tasks = new ArrayList<>();
		Map<Integer[], Integer[]> depMap = new HashMap<>();
		tasks.add(DependencyThreadPool.createDependencyTask(new AllocateOutputFrame(this, out)));
		List<DependencyTask<?>> buildTasks = getBuildTasks(in);  // First half is builder build second half is applier build, dependencies already done
		tasks.addAll(buildTasks);
		List<DependencyTask<?>> applyTasks = tokenizerApplier.getApplyTasks(this.internalRepresentation, out);
		if(applyTasks.size() != buildTasks.size() / 2)
			throw new DMLRuntimeException("Different block sizes between build and apply tasks currently not supported");
		// Builder creates internal representation for a given section
		// Applier builder creates additional meta information which will be needed in the apply step
		// If there is long representation and no padding:
		//  - Count and Hash apply tasks have dependencies to the metadata build task of all previous chunks due to "getOutputRow".
		//	e.g. apply task starting at row 100 with block size 50 has dependencies to the ApplierBuildTask responsible for sections [0-49] and [50-99].
		//  - Same for Position only they are only dependent on the internal representation creation since it does not have metadata.
		if(!tokenizerApplier.isWideFormat() || !tokenizerApplier.hasPadding()){
			int buildTaskOffset;
			if(tokenizerApplier instanceof TokenizerApplierPosition){
				buildTaskOffset = 0;
			}
			else if (tokenizerApplier instanceof TokenizerApplierCount || tokenizerApplier instanceof TokenizerApplierHash) {
				buildTaskOffset = applyTasks.size();
			}
			else{
				throw new DMLRuntimeException("Unknown TokenizerApplier");
			}
			depMap.put(new Integer[] {0, 1}, new Integer[]{1, (buildTasks.size()/2) + 1});
			depMap.put(new Integer[] {tasks.size(), tasks.size()+applyTasks.size()}, new Integer[]{0, 1});
			for(int i = 0; i < applyTasks.size(); i++){
				depMap.put(new Integer[] {tasks.size() + i, tasks.size()+applyTasks.size()}, new Integer[]{1+buildTaskOffset + i, 2+buildTaskOffset + i});
			}
		}
		tasks.addAll(applyTasks);
		List<List<? extends Callable<?>>> deps = new ArrayList<>(Collections.nCopies(tasks.size(), null));
		DependencyThreadPool.createDependencyList(tasks, depMap, deps);
		return DependencyThreadPool.createDependencyTasks(tasks, deps);
	}

	public FrameBlock apply(FrameBlock out, int k) {
		int lastRow = -1;
		if(k > 1){
			DependencyThreadPool pool = new DependencyThreadPool(k);
			try{
				List<DependencyTask<?>> taskList = tokenizerApplier.getApplyTasks(this.internalRepresentation, out);
				lastRow = pool.submitAllAndWait(taskList)//
					.stream().map(x -> (Integer) x).max((x,y) -> Integer.compare(x, y)).get();
			}
			catch(ExecutionException | InterruptedException e) {
				LOG.error("MT Tokenizer apply failed");
				e.printStackTrace();
			}
			pool.shutdown();

		}else{
			lastRow = tokenizerApplier.applyInternalRepresentation(this.internalRepresentation, out);
		}
		if(lastRow != out.getNumRows()){
			out = out.slice(0, lastRow - 1, 0, out.getNumColumns() - 1, null);
		}

		return out;
	}

	public List<DependencyTask<?>> getBuildTasks(FrameBlock in){
		List<DependencyTask<?>> tasks = tokenizerBuilder.getTasks(in, this.internalRepresentation);
		List<DependencyTask<?>> applierBuildTaskList = tokenizerApplier.getBuildTasks(this.internalRepresentation);
		if(tasks.size() != applierBuildTaskList.size())
			throw new DMLRuntimeException("Cannot create dependencies for mismatched array sizes");
		tasks.addAll(applierBuildTaskList);
		List<List<? extends Callable<?>>> deps = new ArrayList<>(Collections.nCopies(tasks.size(), null));
		Map<Integer[], Integer[]> depMap = new HashMap<>();
		for(int i = 0; i < tasks.size() / 2; i++){
			depMap.put(new Integer[]{i+applierBuildTaskList.size(), i+applierBuildTaskList.size() + 1}, new Integer[] {i, i+1});
		}
		DependencyThreadPool.createDependencyList(tasks, depMap, deps);
		tasks = DependencyThreadPool.createDependencyTasks(tasks, deps);
		return tasks;
	}

	public void build(FrameBlock in, int k){
		tokenizerApplier.allocateInternalMeta(in.getNumRows());
		if(k > 1){
			DependencyThreadPool pool = new DependencyThreadPool(k);
			try{
				pool.submitAllAndWait(getBuildTasks(in));
			}
			catch(ExecutionException | InterruptedException e) {
				LOG.error("MT Tokenizer build failed");
				e.printStackTrace();
			}
			pool.shutdown();

		}else{
			tokenizerBuilder.createInternalRepresentation(in, this.internalRepresentation);
			tokenizerApplier.build(this.internalRepresentation, 0, -1);
		}
	}


	protected static class AllocateOutputFrame implements Callable<Object>{

		protected final Tokenizer _tokenizer;
		protected final FrameBlock _out;

		protected AllocateOutputFrame(Tokenizer tokenizer,
									  FrameBlock out){
			this._tokenizer = tokenizer;
			this._out = out;
		}

		@Override
		public Object call() throws Exception {
			_out.ensureAllocatedColumns(_tokenizer.getNumRowsEstimate());
			return null;
		}
	}
}
