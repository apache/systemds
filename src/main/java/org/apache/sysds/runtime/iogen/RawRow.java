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

import org.apache.sysds.runtime.matrix.data.Pair;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashSet;

public class RawRow {

	private final String raw;
	private final ArrayList<Integer> numericPositions = new ArrayList<>();
	private final BitSet numericReserved;
	private final String numericRaw;
	private final BitSet reserved;

	public RawRow(String raw) {
		this.raw = raw;
		char[] rawChars = raw.toCharArray();
		StringBuilder sbNumericRaw = new StringBuilder();
		for(int i = 0; i < rawChars.length; i++) {
			char ch = rawChars[i];
			if(Character.isDigit(ch)) {
				sbNumericRaw.append(ch);
				numericPositions.add(i);
			}
		}
		numericReserved = new BitSet(numericPositions.size());
		numericRaw = sbNumericRaw.toString();
		reserved = new BitSet(raw.length());
	}

	public Pair<Integer, Integer> findValue(ValueTrimFormat vtf) {
		if(vtf instanceof ValueTrimFormat.NumberTrimFormat)
			return findValue((ValueTrimFormat.NumberTrimFormat) vtf);

		return null;
	}

	private Pair<Integer, Integer> findValue(ValueTrimFormat.NumberTrimFormat ntf) {
		ArrayList<Pair<Integer, Integer>> unreserved = getUnreservedPositions();
		Pair<Integer, Integer> result = new Pair<>(-1, 0);
		Pair<Integer, Integer> resultNumeric = new Pair<>(-1, 0);
		for(Pair<Integer, Integer> p : unreserved) {
			int start = p.getKey();
			int end = p.getValue();
			String ntfString = ntf.getNString();
			int length = ntfString.length();
			int index = numericRaw.indexOf(ntfString, start);
			if(index == -1 || index > end - length + 1)
				continue;

			resultNumeric.setValue(length);
			resultNumeric.setKey(index);
			int startPos = numericPositions.get(index);
			int endPos = numericPositions.get(index + length - 1);
			if(ntf.getActualValue() == Double.parseDouble(ntfString)) {
				result.setKey(startPos);
				result.setValue(length);
			}

			// Choose range of string
			boolean flagD = false;

			// 1. the range contain '.'
			// 2. the range contain none numeric chars. In this condition we should terminate checking
			int d = endPos - startPos - length + 1;
			if(d == 1) {
				for(int i = startPos; i <= endPos; i++) {
					if(raw.charAt(i) == '.') {
						flagD = true;
						length++;
						break;
					}
				}
				if(!flagD)
					continue;
				// Check mapping
				ntfString = raw.substring(startPos, endPos + 1);
				// Actual value contain '.'
				if(ntf.getActualValue() == Double.parseDouble(ntfString)) {
					result.setKey(startPos);
					result.setValue(ntfString.length());
				}
			}
			else if(d > 1)
				continue;

			StringBuilder sb = new StringBuilder();

			// 3. if the actual value is positive
			// add some prefixes to the string
			for(int i = startPos - 1; i >= 0; i--) {
				char ch = raw.charAt(i);
				if(ch == '0')
					sb.append(ch);
				else if(!flagD && ch == '.') {
					sb.append(ch);
					flagD = true;
				}
				else if(ch == '+' || ch == '-') {
					sb.append(ch);
					break;
				}
				else
					break;
			}
			sb = sb.reverse();
			startPos -= sb.length();
			sb.append(ntfString);
			if(ntf.getActualValue() == Double.parseDouble(sb.toString())) {
				result.setKey(startPos);
				result.setValue(sb.length());
			}

			// 4. if the actual value is negative '.' can be at the right side of the position
			for(int i = endPos + 1; i < raw.length(); i++) {
				char ch = raw.charAt(i);
				sb.append(ch);
				if(ch == 'E' || ch == 'e' || ch == '+' || ch == '-') {
					continue;
				}
				else if(!Character.isDigit(ch) && ch != '.')
					break;
				Double value = tryParse(sb.toString());
				if(value != null) {
					if(value == ntf.getActualValue()) {
						result.setKey(startPos);
						result.setValue(sb.length());
					}
				}
			}
			if(result.getKey() != -1) {
				break;
			}
		}
		if(result.getKey() != -1) {
			for(int i = resultNumeric.getKey()-1; i >=0 ; i--) {
				if(numericPositions.get(i)>=result.getKey())
					numericReserved.set(i);
				else
					break;
			}

			for(int i = resultNumeric.getKey()+1; i <numericPositions.size() ; i++) {
				if(numericPositions.get(i)<=result.getKey()+result.getValue())
					numericReserved.set(i);
				else
					break;
			}

			numericReserved.set(resultNumeric.getKey(), resultNumeric.getKey() + resultNumeric.getValue(), true);
			reserved.set(result.getKey(), result.getKey() + result.getValue(), true);
		}
		return result;
	}

	private ArrayList<Pair<Integer, Integer>> getUnreservedPositions() {
		ArrayList<Pair<Integer, Integer>> result = new ArrayList<>();
		int sIndex, eIndex;
		int size = numericPositions.size();
		for(int i = 0; i < size; ) {
			// skip all reserved indexes
			for(int j = i; j < size; j++) {
				if(numericReserved.get(j))
					i++;
				else
					break;
			}
			sIndex = i;
			// Extract unreserved position
			for(int j = i; j < size; j++) {
				if(!numericReserved.get(j))
					i++;
				else
					break;
			}
			eIndex = i;
			if(sIndex < eIndex)
				result.add(new Pair<>(sIndex, eIndex - 1));
		}
		return result;
	}

	private Double tryParse(String input) {
		try {
			return Double.parseDouble(input);
		}
		catch(Exception ex) {
			return null;
		}
	}

	public Pair<String, String> getDelims() {
		Pair<String, String> result = new Pair<>("", "");

		StringBuilder sbAll = new StringBuilder();
		StringBuilder sbPart = new StringBuilder();
		String minToken = "";
		for(int i = 0; i < raw.length(); i++) {
			if(!reserved.get(i)) {
				char ch = raw.charAt(i);
				sbAll.append(ch);
				sbPart.append(ch);
			}
			else {
				if(sbPart.length() == 0)
					continue;

				if(minToken.length() == 0 || minToken.length() > sbPart.length())
					minToken = sbPart.toString();

				sbPart = new StringBuilder();
			}
		}
		result.set(minToken, sbAll.toString());
		return result;
	}

	public void resetReserved() {
		reserved.set(0, raw.length(), false);
		numericReserved.set(0, numericPositions.size(), false);
	}

	public boolean isMarked() {
		return !reserved.isEmpty();
	}

	public Pair<HashSet<String>, Integer> getDelimsSet() {
		Pair<HashSet<String>, Integer> result = new Pair<>();
		StringBuilder sb = new StringBuilder();
		int minSize = -1;
		HashSet<String> set = new HashSet<>();
		for(int i = 0; i < raw.length(); i++) {
			if(!reserved.get(i)) {
				char ch = raw.charAt(i);
				sb.append(ch);
			}
			else {
				if(sb.length() > 0) {
					set.add(sb.toString());
					minSize = minSize == -1 ? sb.length() : Math.min(minSize, sb.length());
				}
				sb = new StringBuilder();
			}
		}
		result.set(set, minSize);
		return result;
	}

	public String getRaw() {
		return raw;
	}
}
