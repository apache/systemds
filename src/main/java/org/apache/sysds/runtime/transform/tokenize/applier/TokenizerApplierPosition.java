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
import org.apache.sysds.runtime.matrix.data.FrameBlock;

import org.apache.sysds.runtime.transform.tokenize.Tokenizer;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.apache.sysds.runtime.util.UtilFunctions.getEndIndex;

public class TokenizerApplierPosition extends TokenizerApplier {

	private static final long serialVersionUID = 3563407270742660830L;

	public TokenizerApplierPosition(int numIdCols, int maxTokens, boolean wideFormat) {
		super(numIdCols, maxTokens, wideFormat);
	}

	@Override
	public void applyInternalRepresentation(Tokenizer.DocumentRepresentation[] internalRepresentation, FrameBlock out, int inputRowStart, int blk) {
		int endIndex = getEndIndex(internalRepresentation.length, inputRowStart, blk);
		int outputRow = wideFormat ? inputRowStart : Arrays.stream(internalRepresentation).limit(inputRowStart).mapToInt(doc -> doc.tokens.size()).sum();
		for(int i = inputRowStart; i < endIndex; i++ ) {
			List<Object> keys = internalRepresentation[i].keys;
			List<Tokenizer.Token> tokenList = internalRepresentation[i].tokens;

			if (wideFormat) {
				outputRow = this.appendTokensWide(outputRow, keys, tokenList, out);
			} else {
				outputRow = this.appendTokensLong(outputRow, keys, tokenList, out);
			}
		}
	}


	public int appendTokensLong(int row, List<Object> keys, List<Tokenizer.Token> tokenList, FrameBlock out) {
		int numTokens = 0;
		for (Tokenizer.Token token: tokenList) {
			if (numTokens >= maxTokens) {
				break;
			}
			int col = 0;
			for(; col < keys.size(); col++){
				out.set(row, col, keys.get(col));
			}
			out.set(row, col, token.startIndex + 1);
			out.set(row, col + 1, token.textToken);
			row++;
			numTokens++;
		}
		return row;
	}

	public int appendTokensWide(int row, List<Object> keys, List<Tokenizer.Token> tokenList, FrameBlock out) {
		// Create one row with keys as prefix
		int col = 0;
		for(; col < keys.size(); col++){
			out.set(row, col, keys.get(col));
		}
		int token = 0;
		for (; token < tokenList.size(); token++) {
			if (token >= maxTokens) {
				break;
			}
			out.set(row, col+token, tokenList.get(token).textToken);
		}
		// Remaining positions need to be filled with empty tokens
		for (; token < maxTokens; token++) {
			out.set(row, col+token, "");
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
		Types.ValueType[] schema = UtilFunctions.nCopies(numIdCols + maxTokens,Types.ValueType.STRING );
		return schema;
	}

	private static Types.ValueType[] getOutSchemaLong(int numIdCols) {
		Types.ValueType[] schema = UtilFunctions.nCopies(numIdCols + 2,Types.ValueType.STRING );
		schema[numIdCols] = Types.ValueType.INT64;
		return schema;
	}
}
