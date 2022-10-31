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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.transform.tokenize.DocumentRepresentation;
import org.apache.sysds.runtime.transform.tokenize.Token;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import java.util.ArrayList;
import java.util.List;

import static org.apache.sysds.runtime.util.UtilFunctions.getEndIndex;

public class TokenizerBuilderNgram extends TokenizerBuilderWhitespaceSplit {

	private static final long serialVersionUID = -6297904316677723802L;

	private enum NgramType{
		DOCUMENT,
		TOKEN
	}

	public int minGram = 1;
	public int maxGram = 2;
	public NgramType ngramType = NgramType.DOCUMENT;

	public TokenizerBuilderNgram(int[] idCols, int tokenizeCol, JSONObject params) throws JSONException {
		super(idCols, tokenizeCol, params);
		if (params != null && params.has("min_gram")) {
			this.minGram = params.getInt("min_gram");
		}
		if (params != null && params.has("max_gram")) {
			this.maxGram = params.getInt("max_gram");
		}
		if (params != null && params.has("ngram_type")){
			String type = params.getString("ngram_type").toLowerCase();
			if(type.equals("document")){
				this.ngramType = NgramType.DOCUMENT;
			} else if (type.equals("token")) {
				this.ngramType = NgramType.TOKEN;
			}else {
				throw new DMLRuntimeException("Invalid ngram type, choose between 'token' and 'document'");
			}
		}
	}

	public List<Token> splitIntoNgrams(Token token, int minGram, int maxGram){
		if(token.getNumSubTokens() == 0)
			throw new DMLRuntimeException("Cannot create ngram of token where there are no subTokens");
		if(token.getNumSubTokens() != 1)
			throw new DMLRuntimeException("Cannot create ngram of token where there are more than 1 subTokens");
		String tokenText = token.toString();
		List<Token> newTokens = new ArrayList<>();
		for(int n = minGram; n <= maxGram; n++){
			for(int i = 0; i < tokenText.length() - n + 1; i++){
				String substring = tokenText.substring(i, i+n);
				newTokens.add(new Token(substring, token.getStartIndex(0) + i));
			}
		}
		return newTokens;
	}
	
	@Override
	public void createInternalRepresentation(FrameBlock in, DocumentRepresentation[] internalRepresentation, int rowStart, int blk) {
		super.createInternalRepresentation(in, internalRepresentation, rowStart, blk);
		int endIndex = getEndIndex(in.getNumRows(), rowStart, blk);
		for(int row = rowStart; row < endIndex; row++){
			DocumentRepresentation documentRepresentation = internalRepresentation[row];

			if(this.ngramType == NgramType.DOCUMENT){
				documentRepresentation.splitIntoNgrams(this.minGram, this.maxGram);
			} else if (this.ngramType == NgramType.TOKEN) {
				List<Token> newTokens = new ArrayList<>();
				for (Token wordToken: documentRepresentation.getTokens()) {
					newTokens.addAll(splitIntoNgrams(wordToken, this.minGram, this.maxGram));
				}
				documentRepresentation.tokens = newTokens;
			}
		}
	}
}
