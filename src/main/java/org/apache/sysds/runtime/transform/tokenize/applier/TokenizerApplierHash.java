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

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.transform.tokenize.DocumentRepresentation;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.apache.sysds.runtime.util.UtilFunctions.getEndIndex;

public class TokenizerApplierHash extends TokenizerApplier {

	private static final long serialVersionUID = 4763889041868044668L;

	public int num_features = 1048576;  // 2^20

	private List<Map<Integer, Long>> hashes;

	public TokenizerApplierHash( int numIdCols, int maxTokens, boolean wideFormat, boolean applyPadding, JSONObject params) throws JSONException {
		super(numIdCols, maxTokens, wideFormat, applyPadding);
		if(!applyPadding && wideFormat){
			LOG.warn("ApplyPadding was set to 'false', Hash Tokenizer with wide format always has padding applied");
		}
		if (params != null && params.has("num_features")) {
			this.num_features = params.getInt("num_features");
		}
	}


	@Override
	public int getNumRows(DocumentRepresentation[] internalRepresentation) {
		if(wideFormat)
			return internalRepresentation.length;
		if(applyPadding)
			return maxTokens * internalRepresentation.length;
		return hashes.stream().mapToInt(hashMap -> Math.min(hashMap.size(), maxTokens)).sum();
	}

	@Override
	public void allocateInternalMeta(int numDocuments) {
		hashes = new ArrayList<>(Collections.nCopies(numDocuments,null));
	}

	@Override
	public void build(DocumentRepresentation[] internalRepresentation, int inputRowStart, int blk){
		int endIndex = getEndIndex(internalRepresentation.length, inputRowStart, blk);
		for(int i = inputRowStart; i < endIndex; i++){
			List<Integer> hashList = internalRepresentation[i].tokens.stream().map(token -> {
				int mod = (token.hashCode() % this.num_features);
				if(mod < 0)
					mod += this.num_features;
				return mod;
			}).collect(Collectors.toList());
			Map<Integer, Long> hashCounts = hashList.stream().collect(Collectors.groupingBy(Function.identity(),
					Collectors.counting()));
			hashes.set(i, new TreeMap<>(hashCounts));
		}
	}




	@Override
	public int applyInternalRepresentation(DocumentRepresentation[] internalRepresentation, FrameBlock out, int inputRowStart, int blk) {
		int endIndex = getEndIndex(internalRepresentation.length, inputRowStart, blk);
		int outputRow = getOutputRow(inputRowStart, hashes);
		for(int i = inputRowStart; i < endIndex; i++) {
			List<Object> keys = internalRepresentation[i].keys;
			Map<Integer, Long> sortedHashes = hashes.get(i);
			if (wideFormat) {
				outputRow = this.setTokensWide(outputRow, keys, sortedHashes, out);
			} else {
				outputRow = this.setTokensLong(outputRow, keys, sortedHashes, out);
			}
		}
		return outputRow;
	}


	private int setTokensLong(int row, List<Object> keys, Map<Integer, Long> sortedHashes, FrameBlock out) {
		int numTokens = 0;
		for (Map.Entry<Integer, Long> hashCount: sortedHashes.entrySet()) {
			if (numTokens >= maxTokens) {
				break;
			}
			int col = setKeys(row, keys, out);
			// Create a row per token
			int hash = hashCount.getKey() + 1;
			long count = hashCount.getValue();
			out.set(row, col, (long)hash);
			out.set(row, col + 1, count);
			numTokens++;
			row++;
		}
		if(applyPadding){
			row = applyPaddingLong(row, numTokens, keys, out, PADDING_STRING, 0L);
		}
		return row;
	}

	private int setTokensWide(int row, List<Object> keys, Map<Integer, Long> sortedHashes, FrameBlock out) {
		// Create one row with keys as prefix
		int numKeys = setKeys(row, keys, out);
		for (int tokenPos = 0; tokenPos < maxTokens; tokenPos++) {
			long positionHash = sortedHashes.getOrDefault(tokenPos, 0L);
			out.set(row, numKeys + tokenPos, positionHash);
		}
		return ++row;
	}

	@Override
	public Types.ValueType[] getOutSchema() {
		if (wideFormat) {
			return getOutSchemaWide(numIdCols, maxTokens);
		} else {
			return getOutSchemaLong(numIdCols);
		}
	}

	private static Types.ValueType[] getOutSchemaWide(int numIdCols, int maxTokens) {
		Types.ValueType[] schema = new Types.ValueType[numIdCols + maxTokens];
		int i = 0;
		for (; i < numIdCols; i++) {
			schema[i] = Types.ValueType.STRING;
		}
		for (int j = 0; j < maxTokens; j++, i++) {
			schema[i] = Types.ValueType.INT64;
		}
		return schema;
	}

	private static Types.ValueType[] getOutSchemaLong(int numIdCols) {
		Types.ValueType[] schema =  UtilFunctions.nCopies(numIdCols + 2,Types.ValueType.STRING );
		schema[numIdCols] = Types.ValueType.INT64;
		schema[numIdCols+1] = Types.ValueType.INT64;
		return schema;
	}
}
