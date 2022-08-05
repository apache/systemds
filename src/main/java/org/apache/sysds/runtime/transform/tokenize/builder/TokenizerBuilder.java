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

package org.apache.sysds.runtime.transform.tokenize.builder;

import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.transform.tokenize.DocumentRepresentation;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.runtime.util.DependencyThreadPool;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

import static org.apache.sysds.runtime.transform.tokenize.Tokenizer.TOKENIZE_NUM_BLOCKS;
import static org.apache.sysds.runtime.util.UtilFunctions.getBlockSizes;

public abstract class TokenizerBuilder implements Serializable {
	private static final long serialVersionUID = -4999630313246644464L;

	public void createInternalRepresentation(FrameBlock in, DocumentRepresentation[] internalRepresentation) {
		createInternalRepresentation(in, internalRepresentation, 0, -1);
	}

	public abstract void createInternalRepresentation(FrameBlock in, DocumentRepresentation[] internalRepresentation, int rowStart, int blk);

	public List<DependencyTask<?>> getTasks(FrameBlock in, DocumentRepresentation[] internalRepresentation) {
		int nRows = in.getNumRows();
		List<Callable<Object>> tasks = new ArrayList<>();
		int[] blockSizes = getBlockSizes(nRows, TOKENIZE_NUM_BLOCKS);
		if(blockSizes.length == 1){
			tasks.add(new TokenizerBuildTask<>(this, in, internalRepresentation, 0, -1));
		}
		else {
			for(int startRow = 0, i = 0; i < blockSizes.length; startRow+=blockSizes[i], i++){
			   tasks.add(new TokenizerBuildTask<>(this, in, internalRepresentation, startRow, blockSizes[i]));
			}
		}
		return DependencyThreadPool.createDependencyTasks(tasks, null);
	}


	protected static class TokenizerBuildTask<T extends TokenizerBuilder> implements Callable<Object>{

		protected final T _tokenizerBuilder;
		protected final FrameBlock _input;
		protected final DocumentRepresentation[] _internalRepresentation;
		protected final int _rowStart;
		protected final int _blk;

		protected TokenizerBuildTask(T tokenizerBuilder, FrameBlock input,
									 DocumentRepresentation[] internalRepresentation,
									 int rowStart, int blk){
			this._tokenizerBuilder = tokenizerBuilder;
			this._input = input;
			this._internalRepresentation = internalRepresentation;
			this._rowStart = rowStart;
			this._blk = blk;
		}

		@Override
		public Object call() throws Exception {
			this._tokenizerBuilder.createInternalRepresentation(this._input, this._internalRepresentation, this._rowStart, this._blk);
			return null;
		}
	}
}
