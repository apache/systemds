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

package org.apache.sysds.runtime.util;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

public class DMVUtils {
	public static final char DIGIT = 'd';
	public static final char LOWER = 'l';
	public static final char UPPER = 'u';
	public static final char ALPHA = 'a';
	public static final char SPACE = 's';
	public static final char DOT = 't';
	public static final char OTHER = 'y';
	public static final char ARBITRARY_LEN = '+';
	public static final char MINUS = '-';

	public enum LEVEL_ENUM { LEVEL1, LEVEL2, LEVEL3, LEVEL4, LEVEL5, LEVEL6}

	public static FrameBlock syntacticalPatternDiscovery(FrameBlock frame, double threshold, String disguised_value) {
		// Preparation
		String disguisedVal = disguised_value;
		int numCols = frame.getNumColumns();
		int numRows = frame.getNumRows();
		ArrayList<Map<String, Integer>> table_Hist = new ArrayList<>(numCols); // list of every column with values and their frequency

		for (int idx = 0; idx < numCols; idx++) {
			Object c = frame.getColumnData(idx);
			String[] attr = (String[]) c;
			String key = "";
			for (int i = 0;  i < numRows; i++) {
				key = (attr[i] == null) ? "NULL": attr[i];
				addDistinctValueOrIncrementCounter(table_Hist, key, idx);
			}
		}

		// Syntactic Pattern Discovery
		int idx = -1;
		for (Map<String, Integer> col_Hist : table_Hist) {
			idx++;
			Map<String, Double> dominant_patterns_ratio = new HashMap<>();
			Map<String, Integer> prev_pattern_hist = col_Hist;
			for(LEVEL_ENUM level : LEVEL_ENUM.values()) {
				dominant_patterns_ratio.clear();
				Map<String, Integer> current_pattern_hist = LevelsExecutor(prev_pattern_hist, level);
				dominant_patterns_ratio = calculatePatternsRatio(current_pattern_hist, numRows);
				String dominant_pattern = findDominantPattern(dominant_patterns_ratio, threshold);
				if(dominant_pattern != null) { //found pattern
					detectDisguisedValues(dominant_pattern, frame.getColumnData(idx), idx, frame, level, disguisedVal);
					break;
				}
				prev_pattern_hist = current_pattern_hist;
			}
		}
		return frame;
	}


	public static Map<String, Double> calculatePatternsRatio(Map<String, Integer> patterns_hist, int nr_entries) {
		Map<String, Double> patterns_ratio_map = new HashMap<>();
		for(Entry<String, Integer> e : patterns_hist.entrySet()) {
			String pattern = e.getKey();
			double nr_occurences = e.getValue();
			double current_ratio = nr_occurences / nr_entries; // percentage of current pattern in column
			patterns_ratio_map.put(pattern, current_ratio);
		}
		return patterns_ratio_map;
	}

	public static String findDominantPattern(Map<String, Double> dominant_patterns, double threshold) {
		for(Entry<String, Double> e : dominant_patterns.entrySet()) {
			String pattern = e.getKey();
			Double pattern_ratio = e.getValue();
			if(pattern_ratio > threshold)
				return pattern;
		}
		return null;
	}

	private static void addDistinctValueOrIncrementCounter(ArrayList<Map<String, Integer>> maps, String key, Integer idx) {
		if (maps.size() == idx) {
			HashMap<String, Integer> m = new HashMap<>();
			m.put(key, 1);
			maps.add(m);
			return;
		}

		if (!(maps.get(idx).containsKey(key)))
			maps.get(idx).put(key, 1);
		else
			maps.get(idx).compute(key, (k, v) -> v + 1);
	}

	private static void addDistinctValueOrIncrementCounter(Map<String, Integer> map, String encoded_value, Integer nr_occurrences) {
		if (!(map.containsKey(encoded_value)))
			map.put(encoded_value, nr_occurrences);
		else
			map.compute(encoded_value, (k, v) -> v + nr_occurrences);
	}

	public static Map<String, Integer> LevelsExecutor(Map<String, Integer> old_pattern_hist, LEVEL_ENUM level) {
		Map<String, Integer> new_pattern_hist = new HashMap<>();
		for(Entry<String, Integer> e : old_pattern_hist.entrySet()) {
			String pattern = e.getKey();
			Integer nr_of_occurrences = e.getValue();

			String new_pattern;
			switch(level) {
				case LEVEL1: // default encoding
					new_pattern = encodeRawString(pattern);
					break;
				case LEVEL2: // ignores the number of occurrences. It replaces all numbers with '+'
					new_pattern = removeNumbers(pattern);
					break;
				case LEVEL3: // ignores upper and lowercase characters. It replaces all 'u' and 'l' with 'a' = Alphabet
					new_pattern = removeUpperLowerCase(pattern);
					break;
				case LEVEL4: // changes floats to digits
					new_pattern = removeInnerCharacterInPattern(pattern, DIGIT, DOT);
					break;
				case LEVEL5: // removes spaces between strings
					new_pattern = removeInnerCharacterInPattern(pattern, ALPHA, SPACE);
					break;
				case LEVEL6: // changes negative numbers to digits
					new_pattern = acceptNegativeNumbersAsDigits(pattern);
					break;
				default:
					new_pattern = "";
					break;
			}
			addDistinctValueOrIncrementCounter(new_pattern_hist, new_pattern, nr_of_occurrences);
		}

		return new_pattern_hist;
	}

	public static String acceptNegativeNumbersAsDigits(String pattern) {
		char[] chars = pattern.toCharArray();
		StringBuilder tmp = new StringBuilder();
		boolean currently_minus_digit = false;
		for (char ch : chars) {
			if(ch == MINUS && !currently_minus_digit) {
				currently_minus_digit = true;
			}
			else if(ch == DIGIT && currently_minus_digit) {
				tmp.append(ch);
				currently_minus_digit = false;
			}
			else if(currently_minus_digit) {
				tmp.append(MINUS);
				tmp.append(ch);
				currently_minus_digit = false;
			}
			else {
				tmp.append(ch);
			}
		}
		return tmp.toString();
	}

	public static String removeInnerCharacterInPattern(String pattern, char outter_char, char inner_char) {
		char[] chars = pattern.toCharArray();
		StringBuilder tmp = new StringBuilder();
		boolean currently_digit = false;
		for (char ch : chars) {
			if(ch == outter_char && !currently_digit) {
				currently_digit = true;
				tmp.append(ch);
			}
			else if(currently_digit && (ch == outter_char || ch == inner_char))
				continue;
			else if(ch != inner_char && ch != ARBITRARY_LEN) {
				currently_digit = false;
				tmp.append(ch);
			}
			else {
				if(tmp.length() > 0 && tmp.charAt(tmp.length() - 1) != ARBITRARY_LEN)
					tmp.append(ch);
			}
		}
		return tmp.toString();
	}

	public static String removeUpperLowerCase(String pattern) {
		char[] chars = pattern.toCharArray();
		StringBuilder tmp = new StringBuilder();
		boolean currently_alphabetic = false;
		for (char ch : chars) {
			if(ch == UPPER || ch == LOWER) {
				if(!currently_alphabetic) {
					currently_alphabetic = true;
					tmp.append(ALPHA);
				}
			}
			else if(ch == ARBITRARY_LEN) {
				if(tmp.charAt(tmp.length() - 1) != ARBITRARY_LEN)
					tmp.append(ch);
			}
			else {
				tmp.append(ch);
				currently_alphabetic = false;
			}
		}
		return tmp.toString();
	}

	private static String removeNumbers(String pattern) {
		char[] chars = pattern.toCharArray();
		StringBuilder tmp = new StringBuilder();
		for (char ch : chars) {
			if(Character.isDigit(ch))
				tmp.append(ARBITRARY_LEN);
			else
				tmp.append(ch);
		}
		return tmp.toString();
	}

	public static String encodeRawString(String input) {
		char[] chars = input.toCharArray();

		StringBuilder tmp = new StringBuilder();
		for (char ch : chars) {
			tmp.append(getCharClass(ch));
		}
		return getFrequencyOfEachConsecutiveChar(tmp.toString());
	}

	private static char getCharClass(char c) {
		if (Character.isDigit(c)) return DIGIT;
		if (Character.isLowerCase(c)) return LOWER;
		if (Character.isUpperCase(c)) return UPPER;
		if (Character.isSpaceChar(c)) return SPACE;
		if (c == '.') return DOT;
		if(c == '-') return MINUS;
		return OTHER;
	}

	public static String getFrequencyOfEachConsecutiveChar(String s) {
		StringBuilder retval = new StringBuilder();
		for (int i = 0; i < s.length(); i++) {
			int count = 1;
			while (i + 1 < s.length() && s.charAt(i) == s.charAt(i + 1)) {
				i++;
				count++;
			}
			retval.append(s.charAt(i));
			retval.append(count);
		}
		return retval.toString();
	}

	private static void detectDisguisedValues(String dom_pattern, Object col, int col_idx,
		FrameBlock frameBlock, LEVEL_ENUM level, String disguisedVal)
	{
		int row_idx = -1;
		String pattern = "";
		String[] attr = (String[]) col;
		int numRows = frameBlock.getNumRows();
		for (int i = 0; i < numRows; i++) {
			String value = (attr[i] == null)? "NULL": attr[i];
			switch (level){
				case LEVEL1:
					pattern = encodeRawString(value);
					break;
				case LEVEL2:
					pattern = encodeRawString(value);
					pattern = removeNumbers(pattern);
					break;
				case LEVEL3:
					pattern = encodeRawString(value);
					pattern = removeNumbers(pattern);
					pattern = removeUpperLowerCase(pattern);
					break;
				case LEVEL4:
					pattern = encodeRawString(value);
					pattern = removeNumbers(pattern);
					pattern = removeUpperLowerCase(pattern);
					pattern = removeInnerCharacterInPattern(pattern, DIGIT, DOT);
					break;
				case LEVEL5:
					pattern = encodeRawString(value);
					pattern = removeNumbers(pattern);
					pattern = removeUpperLowerCase(pattern);
					pattern = removeInnerCharacterInPattern(pattern, DIGIT, DOT);
					pattern = removeInnerCharacterInPattern(pattern, ALPHA, SPACE);
					break;
				case LEVEL6:
					pattern = encodeRawString(value);
					pattern = removeNumbers(pattern);
					pattern = removeUpperLowerCase(pattern);
					pattern = removeInnerCharacterInPattern(pattern, DIGIT, DOT);
					pattern = removeInnerCharacterInPattern(pattern, ALPHA, SPACE);
					pattern = acceptNegativeNumbersAsDigits(pattern);
					break;
				default:
					throw new DMLRuntimeException("Could not find suitable level");
			}
			row_idx++;
			if(pattern.equals(dom_pattern))
				continue;
			frameBlock.set(row_idx, col_idx, disguisedVal);
		}
	}
}
