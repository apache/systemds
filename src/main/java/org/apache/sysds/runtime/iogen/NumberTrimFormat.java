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

public class NumberTrimFormat implements Comparable {

	public char S;
	public char[] N;
	public double actualValue;
	public int c;

	public NumberTrimFormat(int c, double value) {
		this(value);
		this.c = c;
	}

	public NumberTrimFormat(double value) {
		actualValue = value;
		if(value == 0) {
			S = '+';
			N = new char[1];
			N[0] = '0';
		}
		else {
			if(value > 0) {
				S = '+';
			}
			else
				S = '-';

			String stringValue = String.format("%.30f", value);
			double d = value - (int) value;
			if(d == 0) {
				stringValue = stringValue.substring(0, stringValue.indexOf("."));
			}

			if(stringValue.charAt(0) == '+' || stringValue.charAt(0) == '-')
				stringValue = stringValue.substring(1);

			int length = stringValue.length();
			int firstNZ = -1;
			int lastNZ = -1;
			for(int i = 0; i < length; i++) {
				char fChar = stringValue.charAt(i);
				char lChar = stringValue.charAt(length - i - 1);
				if(Character.isDigit(fChar) && fChar != '0' && firstNZ == -1)
					firstNZ = i;

				if(Character.isDigit(lChar) && lChar != '0' && lastNZ == -1)
					lastNZ = length - i;

				if(firstNZ > 0 && lastNZ > 0)
					break;
			}
			String subValue = stringValue.substring(firstNZ, lastNZ);
			int dotLength = subValue.contains(".") ? 1 : 0;
			N = new char[lastNZ - firstNZ - dotLength];
			int index = 0;
			for(Character c : subValue.toCharArray()) {
				if(c != '.')
					N[index++] = c;
			}
		}
	}

	public NumberMappingInfo getMappingInfo(String input) {
		return getMappingInfo(input, false);
	}

	public NumberMappingInfo getMappingInfoIncludeZero(String input) {
		return getMappingInfo(input, true);
	}

	private NumberMappingInfo getMappingInfo(String input, boolean enableZero) {
		NumberMappingInfo result = new NumberMappingInfo();
		if(input.length() < N.length)
			return result;

		char[] chunkChars = input.toCharArray();
		int ccIndex = 0;
		int currentIndex = 0;

		while(currentIndex < chunkChars.length && !result.mapped) {

			StringBuilder actualValueChars = new StringBuilder();
			int skip = -1;
			// 1. Choose the signe
			for(int i = currentIndex; i < chunkChars.length; i++) {
				skip++;
				if(S == '-') {
					if(chunkChars[i] == '-') {
						actualValueChars.append('-');
						ccIndex = i + 1;
						break;
					}
				}
				else if(S == '+') {
					actualValueChars.append('+');
					ccIndex = i;
					if(chunkChars[i] == '+') {
						ccIndex++;
					}
					break;
				}
			}

			// 2. Skip all zero and '.' until to find a none-zero value
			int dotPos = -1;
			int firstNZIndex = -1;
			for(int i = ccIndex; i < chunkChars.length; i++) {
				if(chunkChars[i] == '0' || chunkChars[i] == '.') {
					if(chunkChars[i] == '.')
						dotPos = i;
					actualValueChars.append(chunkChars[i]);
				}
				else {
					firstNZIndex = i;
					break;
				}
			}
			// The text value is Zero, i.e., 0.00000, 000000, 000.000
			if(firstNZIndex == -1) {
				if(enableZero) {
					result.size = chunkChars.length;
					result.mapped = true;
					result.index = currentIndex + skip;
					break;
				}
				else
					break;
			}
			if(actualValueChars.length() > 1 && enableZero && isEqual(actualValueChars, actualValue)) {
				result.size = firstNZIndex - currentIndex;
				result.mapped = true;
				result.index = currentIndex + skip;
				break;
			}

			// look for N char list
			result.size = firstNZIndex - currentIndex - skip;
			int currentPos = firstNZIndex;
			boolean NFlag = false;
			for(int i = currentPos, j = 0; i < chunkChars.length && j < N.length; i++) {
				char cc = chunkChars[i];
				char c = N[j];
				if(cc == '.' && dotPos == -1) {
					dotPos = currentIndex + result.size;
					result.size++;
					actualValueChars.append(cc);
					currentPos++;
					continue;
				}
				if(cc == c) {
					result.size++;
					actualValueChars.append(cc);
				}
				else {
					NFlag = false;
					break;
				}
				currentPos++;
				j++;
				if(j == N.length)
					NFlag = true;
			}

			// NM char list matched, So, look for science values or "0" values
			if(NFlag) {
				if(isEqual(actualValueChars, actualValue)) {
					result.mapped = true;
					result.index = currentIndex + skip;
					break;
				}
				else if(currentPos == chunkChars.length)
					break;
				else {
					boolean eFlag = false;
					for(int i = currentPos; i < chunkChars.length; i++) {
						char vChar = Character.toUpperCase(chunkChars[i]);
						if(!eFlag) {
							if(vChar == '.' && dotPos != -1) {
								currentIndex++;
								break;
							}
							else if(vChar == '.' || vChar == '0') {
								actualValueChars.append(vChar);
								result.size++;
								if(isEqual(actualValueChars, actualValue)) {
									result.mapped = true;
									result.index = currentIndex + skip;
									break;
								}
							}
							// check for "E/e"
							else if(vChar == 'E') {
								eFlag = true;
								actualValueChars.append('E');
								result.size++;
								// check for "+/-"
								if((i + 1) < chunkChars.length && (chunkChars[i + 1] == '+' || chunkChars[i + 1] == '-')) {
									actualValueChars.append(chunkChars[i + 1]);
									actualValueChars.append('0');
									i++;
									result.size++;
								}
							}
							else {
								currentIndex++;
								break;
							}
						}
						else {
							if(Character.isDigit(vChar)) {
								actualValueChars.append(vChar);
								result.size++;
								if(isEqual(actualValueChars, actualValue)) {
									result.mapped = true;
									result.index = currentIndex + skip;
									break;
								}
							}
							else {
								break;
							}
						}
					}
				}
			}
			else {
				currentIndex++;
				result.size = -1;
				result.index = -1;
			}
		}
		return result;
	}

	private boolean isEqual(StringBuilder valueChars, double value) {
		return Double.parseDouble(valueChars.toString()) == value;
	}

	@Override public String toString() {
		StringBuilder s = new StringBuilder(S + "");
		for(Character c : N)
			s.append(c);
		return s.toString();
	}

	public NumberTrimFormat getACopy() {
		NumberTrimFormat copy = new NumberTrimFormat(c, actualValue);
		copy.S = S;
		copy.N = N;
		return copy;
	}

	@Override public int compareTo(Object ntf) {
		int vc = Double.compare(Math.abs(((NumberTrimFormat) ntf).actualValue), Math.abs(actualValue));
		int nc = Integer.compare(((NumberTrimFormat) ntf).N.length, N.length);
		int r;

		if(vc == 0)
			return 0;

		if(nc == 1) {
			return 1;
		}
		else if(nc == 0) {
			return vc >= 0 ? 1 : -1;
		}
		else {
			return -1;
		}
	}
}
