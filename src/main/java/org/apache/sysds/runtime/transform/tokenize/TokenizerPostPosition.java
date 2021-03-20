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
import org.apache.wink.json4j.JSONObject;

import java.util.ArrayList;
import java.util.List;

public class TokenizerPostPosition implements TokenizerPost{

	private static final long serialVersionUID = 3563407270742660830L;
	private final int numIdCols;
	private final int maxTokens;
	private final boolean wideFormat;

	public TokenizerPostPosition(JSONObject params, int numIdCols, int maxTokens, boolean wideFormat) {
		// No configurable params yet
		this.numIdCols = numIdCols;
		this.maxTokens = maxTokens;
		this.wideFormat = wideFormat;
	}

	@Override
	public FrameBlock tokenizePost(List<Tokenizer.DocumentToTokens> tl, FrameBlock out) {
		for (Tokenizer.DocumentToTokens docToToken: tl) {
			List<Object> keys = docToToken.keys;
			List<Tokenizer.Token> tokenList = docToToken.tokens;

			if (wideFormat) {
				this.appendTokensWide(keys, tokenList, out);
			} else {
				this.appendTokensLong(keys, tokenList, out);
			}
		}

		return out;
	}

	public void appendTokensLong(List<Object> keys, List<Tokenizer.Token> tokenList, FrameBlock out) {
		int numTokens = 0;
		for (Tokenizer.Token token: tokenList) {
			if (numTokens >= maxTokens) {
				break;
			}
			// Create a row per token
			List<Object> rowList = new ArrayList<>(keys);
			// Convert to 1-based index for DML
			rowList.add(token.startIndex + 1);
			rowList.add(token.textToken);
			Object[] row = new Object[rowList.size()];
			rowList.toArray(row);
			out.appendRow(row);
			numTokens++;
		}
	}

	public void appendTokensWide(List<Object> keys, List<Tokenizer.Token> tokenList, FrameBlock out) {
		// Create one row with keys as prefix
		List<Object> rowList = new ArrayList<>(keys);

		int numTokens = 0;
		for (Tokenizer.Token token: tokenList) {
			if (numTokens >= maxTokens) {
				break;
			}
			rowList.add(token.textToken);
			numTokens++;
		}
		// Remaining positions need to be filled with empty tokens
		for (; numTokens < maxTokens; numTokens++) {
			rowList.add("");
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
		Types.ValueType[] schema = UtilFunctions.nCopies(numIdCols + maxTokens,Types.ValueType.STRING );
		return schema;
	}

	private static Types.ValueType[] getOutSchemaLong(int numIdCols) {
		Types.ValueType[] schema = new Types.ValueType[numIdCols + 2];
		int i = 0;
		for (; i < numIdCols; i++) {
			schema[i] = Types.ValueType.STRING;
		}
		schema[i] = Types.ValueType.INT64;
		schema[i+1] = Types.ValueType.STRING;
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
