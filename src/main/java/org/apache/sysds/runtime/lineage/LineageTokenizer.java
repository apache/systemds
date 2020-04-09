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


package org.apache.sysds.runtime.lineage;

import org.apache.sysds.parser.ParseException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class LineageTokenizer {
	
	private final List<TokenInfo> _tokenInfos;
	
	public LineageTokenizer() {
		_tokenInfos = new ArrayList<>();
	}
	
	public void add(String regex, String key) {
		_tokenInfos.add(new TokenInfo(Pattern.compile("^(" + regex + ")"), key));
	}
	
	public void add(String regex) {
		_tokenInfos.add(new TokenInfo(Pattern.compile("^(" + regex + ")"), ""));
	}
	
	public Map<String, String> tokenize(String str) {
		Map<String, String> tokens = new HashMap<>();
		
		for (TokenInfo info : _tokenInfos) {
			Matcher m = info.regex.matcher(str);
			if (m.find()) {
				String tok = m.group().trim();
				if (!info.key.isEmpty())
					tokens.put(info.key, tok);
				str = m.replaceFirst("");
			}
		}
		if (!str.isEmpty())
			throw new ParseException("Unexpected character in input: " + str);
		return tokens;
	}
	
	private class TokenInfo {
		public final Pattern regex;
		public final String key;
		
		public TokenInfo(Pattern regex, String key) {
			super();
			this.regex = regex;
			this.key = key;
		}
	}
}