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

package org.apache.sysds.runtime.transform.tokenize.applier;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.transform.tokenize.DocumentRepresentation;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.runtime.util.DependencyThreadPool;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

import static org.apache.sysds.runtime.transform.tokenize.Tokenizer.TOKENIZE_NUM_BLOCKS;
import static org.apache.sysds.runtime.util.UtilFunctions.getBlockSizes;

public abstract class TokenizerApplier implements Serializable {
	private static final long serialVersionUID = 39116559705096787L;

	protected static final Log LOG = LogFactory.getLog(TokenizerApplier.class.getName());

	public static final String PADDING_STRING = "";

	protected final int numIdCols;
	protected final int maxTokens;
	protected final boolean wideFormat;
	protected final boolean applyPadding;

	public TokenizerApplier(int numIdCols, int maxTokens, boolean wideFormat, boolean applyPadding){
		this.numIdCols = numIdCols;
		this.maxTokens = maxTokens;
		this.wideFormat = wideFormat;
		this.applyPadding = applyPadding;
	}

	public int applyInternalRepresentation(DocumentRepresentation[] internalRepresentation, FrameBlock out){
		return applyInternalRepresentation(internalRepresentation, out, 0, -1);
	}
	abstract int applyInternalRepresentation(DocumentRepresentation[] internalRepresentation, FrameBlock out, int startRow, int blk);

	public void build(DocumentRepresentation[] internalRepresentation, int inputRowStart, int blk){ }

	public List<DependencyTask<?>> getBuildTasks(DocumentRepresentation[] internalRepresentation){
		int nRows = internalRepresentation.length;
		List<Callable<Object>> tasks = new ArrayList<>();
		int[] blockSizes = getBlockSizes(nRows, TOKENIZE_NUM_BLOCKS);
		if(blockSizes.length == 1){
			tasks.add(new TokenizerApplierBuildTask<>(this, internalRepresentation, 0, -1));
		}
		else {
			for(int startRow = 0, i = 0; i < blockSizes.length; startRow+=blockSizes[i], i++){
				tasks.add(new TokenizerApplierBuildTask<>(this, internalRepresentation, startRow, blockSizes[i]));
			}
		}
		return DependencyThreadPool.createDependencyTasks(tasks, null);
	}

	public List<DependencyTask<?>> getApplyTasks(DocumentRepresentation[] internalRepresentation, FrameBlock out) {
		int nRows = internalRepresentation.length;
		List<Callable<Object>> tasks = new ArrayList<>();
		int[] blockSizes = getBlockSizes(nRows, TOKENIZE_NUM_BLOCKS);
		if(blockSizes.length == 1){
			tasks.add(new TokenizerApplyTask<>(this, out, internalRepresentation, 0, -1));
		}
		else {
			for(int startRow = 0, i = 0; i < blockSizes.length; startRow+=blockSizes[i], i++){
				tasks.add(new TokenizerApplyTask<>(this, out, internalRepresentation, startRow, blockSizes[i]));
			}
		}
		return DependencyThreadPool.createDependencyTasks(tasks, null);
	}

	protected int setKeys(int row, List<Object> keys, FrameBlock out){
		int col = 0;
		for(; col < keys.size(); col++){
			out.set(row, col, keys.get(col));
		}
		return col;
	}

	protected int applyPaddingLong(int startRow, int numTokens, List<Object> keys, FrameBlock out, Object val1, Object val2){
		int row = startRow;
		for (; numTokens < maxTokens; numTokens++, row++){
			int col = setKeys(row, keys, out);
			out.set(row, col, val1);
			out.set(row, col+1, val2);
		}
		return row;
	}

	protected void applyPaddingWide(int row, int offset, int startToken, FrameBlock out, Object padding){
		int token = startToken;
		for (; token < maxTokens; token++) {
			out.set(row, offset+token, padding);
		}
	}

	public abstract Types.ValueType[] getOutSchema();

	public boolean hasPadding(){
		return applyPadding;
	}

	public int getMaxTokens(){
		return maxTokens;
	}

	public int getMaxNumRows(int inRows) {
		if (wideFormat) {
			return inRows;
		} else {
			return inRows * maxTokens;
		}
	}
	public abstract int getNumRows(DocumentRepresentation[] internalRepresentation);

	public <T, E> int getOutputRow(int inputRowStart, List<Map<T, E>> internalData){
		if(wideFormat)
			return inputRowStart;
		if(applyPadding)
			return maxTokens * inputRowStart;
		return internalData.stream().limit(inputRowStart).mapToInt(hashMap -> Math.min(hashMap.size(), maxTokens)).sum();
	}

	public int getOutputRow(int inputRowStart, DocumentRepresentation[] internalData){
		if(wideFormat)
			return inputRowStart;
		if(applyPadding)
			return maxTokens * inputRowStart;
		return Arrays.stream(internalData).limit(inputRowStart).mapToInt(doc -> Math.min(doc.tokens.size(), maxTokens)).sum();
	}

	public long getNumCols() {
		return this.getOutSchema().length;
	}

	public boolean isWideFormat() {
		return wideFormat;
	}

	public void allocateInternalMeta(int numDocuments) { }


	protected static class TokenizerApplyTask<T extends TokenizerApplier> implements Callable<Object>{

		protected final T _tokenizerApplier;
		protected final FrameBlock _output;
		protected final DocumentRepresentation[] _internalRepresentation;
		protected final int _rowStart;
		protected final int _blk;

		protected TokenizerApplyTask(T tokenizerApplier, FrameBlock out,
									 DocumentRepresentation[] internalRepresentation,
									 int rowStart, int blk){
			this._tokenizerApplier = tokenizerApplier;
			this._output = out;
			this._internalRepresentation = internalRepresentation;
			this._rowStart = rowStart;
			this._blk = blk;
		}

		@Override
		public Object call() throws Exception {
			return this._tokenizerApplier.applyInternalRepresentation(this._internalRepresentation, this._output, this._rowStart, this._blk);
		}
	}

	protected static class TokenizerApplierBuildTask<T extends TokenizerApplier> implements Callable<Object>{

		protected final T _tokenizerApplier;
		protected final DocumentRepresentation[] _internalRepresentation;
		protected final int _rowStart;
		protected final int _blk;

		protected TokenizerApplierBuildTask(T tokenizerApplier,
				DocumentRepresentation[] internalRepresentation, int rowStart, int blk){
			_tokenizerApplier = tokenizerApplier;
			_internalRepresentation = internalRepresentation;
			_rowStart = rowStart;
			_blk = blk;
		}

		@Override
		public Object call() throws Exception {
			this._tokenizerApplier.build(this._internalRepresentation, this._rowStart, this._blk);
			return null;
		}
	}
}
