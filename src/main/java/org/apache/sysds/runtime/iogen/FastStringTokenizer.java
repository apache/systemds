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

package org.apache.sysds.runtime.iogen;

import java.io.Serializable;
import java.util.HashSet;

public class FastStringTokenizer implements Serializable {

	private static final long serialVersionUID = -4698672725609750097L;
	private String _string = null;
	private String _del = "";
	private int _pos = -1;
	private int _index = 0;
	private HashSet<String> naStrings = null;

	public FastStringTokenizer(String delimiter) {
		_del = delimiter;
		reset(null);
	}

	public void reset(String string) {
		_string = string;
		_pos = 0;
		_index = 0;
	}

	public String nextToken() {
		int len = _string.length();
		int start = _pos;

		if(_pos == -1) {
			_index = -1;
			return "0";
		}
		//find start (skip over leading delimiters)
		while(start < len && _del.equals(_string.substring(start, Math.min(start + _del.length(), _string.length())))) {
			start += _del.length();
			_index++;
		}

		//find end (next delimiter) and return
		if(start < len) {
			_pos = _string.indexOf(_del, start);
			if(start < _pos && _pos < len)
				return _string.substring(start, _pos);
			else
				return _string.substring(start);
		}
		//no next token
		_index = -1;
		return null;
	}

	public int nextInt() {
		return Integer.parseInt(nextToken());
	}

	public long nextLong() {
		return Long.parseLong(nextToken());
	}

	public double nextDouble() {
		String nt = nextToken();
		if((naStrings != null && naStrings.contains(nt)) || nt == null)
			return 0;
		else
			return Double.parseDouble(nt);
	}

	public int getIndex() {
		return _index;
	}

	public void setNaStrings(HashSet<String> naStrings) {
		this.naStrings = naStrings;
	}
}
