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

import java.util.ArrayList;
import java.util.List;

public class DocumentRepresentation {
	public List<Object> keys;
	public List<Token> tokens;

	public DocumentRepresentation(List<Object> keys, List<Token> tokens) {
		this.keys = keys;
		this.tokens = tokens;
	}

	public List<Token> getTokens() {
		return tokens;
	}


	public void splitIntoNgrams(int minGram, int maxGram){
		List<Token> ngramTokens = new ArrayList<>();
		for(int n = minGram; n <= maxGram; n++){
			for(int i = 0; i < tokens.size() - n + 1; i++){
				List<Token> subList = tokens.subList(i, i+n);
				Token token = new Token(subList);
				ngramTokens.add(token);
			}
		}
		tokens = ngramTokens;
	}
}
