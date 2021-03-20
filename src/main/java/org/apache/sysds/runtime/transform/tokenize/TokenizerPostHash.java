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

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Function;
import java.util.stream.Collectors;

public class TokenizerPostHash implements TokenizerPost{

	private static final long serialVersionUID = 4763889041868044668L;
	private final Params params;
	private final int numIdCols;
	private final int maxTokens;
	private final boolean wideFormat;

	static class Params implements Serializable {

		private static final long serialVersionUID = -256069061414241795L;

		public int num_features = 1048576;  // 2^20

		public Params(JSONObject json) throws JSONException {
			if (json != null && json.has("num_features")) {
				this.num_features = json.getInt("num_features");
			}
		}
	}

	public TokenizerPostHash(JSONObject params, int numIdCols, int maxTokens, boolean wideFormat) throws JSONException {
		this.params = new Params(params);
		this.numIdCols = numIdCols;
		this.maxTokens = maxTokens;
		this.wideFormat = wideFormat;
	}

	@Override
	public FrameBlock tokenizePost(List<Tokenizer.DocumentToTokens> tl, FrameBlock out) {
		for (Tokenizer.DocumentToTokens docToToken: tl) {
			List<Object> keys = docToToken.keys;
			List<Tokenizer.Token> tokenList = docToToken.tokens;
			// Transform to hashes
			List<Integer> hashList = tokenList.stream().map(token -> token.textToken.hashCode() %
				params.num_features).collect(Collectors.toList());
			// Counting the hashes
			Map<Integer, Long> hashCounts = hashList.stream().collect(Collectors.groupingBy(Function.identity(),
				Collectors.counting()));
			// Sorted by hash
			Map<Integer, Long> sortedHashes = new TreeMap<>(hashCounts);

			if (wideFormat) {
				this.appendTokensWide(keys, sortedHashes, out);
			} else {
				this.appendTokensLong(keys, sortedHashes, out);
			}
		}

		return out;
	}

	private void appendTokensLong(List<Object> keys, Map<Integer, Long> sortedHashes, FrameBlock out) {
		int numTokens = 0;
		for (Map.Entry<Integer, Long> hashCount: sortedHashes.entrySet()) {
			if (numTokens >= maxTokens) {
				break;
			}
			// Create a row per token
			int hash = hashCount.getKey() + 1;
			long count = hashCount.getValue();
			List<Object> rowList = new ArrayList<>(keys);
			rowList.add((long) hash);
			rowList.add(count);
			Object[] row = new Object[rowList.size()];
			rowList.toArray(row);
			out.appendRow(row);
			numTokens++;
		}
	}

	private void appendTokensWide(List<Object> keys, Map<Integer, Long> sortedHashes, FrameBlock out) {
		// Create one row with keys as prefix
		List<Object> rowList = new ArrayList<>(keys);

		for (int tokenPos = 0; tokenPos < maxTokens; tokenPos++) {
			long positionHash = sortedHashes.getOrDefault(tokenPos, 0L);
			rowList.add(positionHash);
		}
		Object[] row = new Object[rowList.size()];
		rowList.toArray(row);
		out.appendRow(row);
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

	public long getNumRows(long inRows) {
		if (wideFormat) {
			return inRows;
		} else {
			return inRows * maxTokens;
		}
	}

	public long getNumCols() {
		return this.getOutSchema().length;
	}
}
