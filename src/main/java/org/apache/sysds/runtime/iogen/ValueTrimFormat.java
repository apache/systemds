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

import org.apache.sysds.common.Types;

public abstract class ValueTrimFormat implements Comparable {

	// save the col index of the value on the Matrix.
	// We need this value when we want to reorder matrix cols
	protected int colIndex;

	public ValueTrimFormat() {
	}

	// Get a copy of value
	public abstract ValueTrimFormat getACopy();

	// Create a VTF based on 3 data types
	public static ValueTrimFormat createNewTrimFormat(Types.ValueType vt, Object in) {
		switch(vt) {
			case STRING:
				return new StringTrimFormat(in.toString());
			case BOOLEAN:
				return new BooleanTrimFormat((Boolean) in);
			default:
				return new NumberTrimFormat((Double) in);
		}
	}

	// Check the value is a not set value
	public abstract boolean isNotSet();

	// Set as NoSet
	public abstract void setNoSet();

	// Get String of actual value
	public abstract String getStringOfActualValue();

	public abstract boolean isEqual(ValueTrimFormat vtf);

	public int getColIndex() {
		return colIndex;
	}

	// Convert String value to trim format
	public static class StringTrimFormat extends ValueTrimFormat {

		private String actualValue; // save the actual value

		public StringTrimFormat(int colIndex, String actualValue) {
			this(actualValue);
			this.colIndex = colIndex;
		}

		public StringTrimFormat(String actualValue) {
			this.actualValue = actualValue;
		}

		@Override public StringTrimFormat getACopy() {
			StringTrimFormat copy = new StringTrimFormat(colIndex, actualValue);
			return copy;
		}

		@Override public boolean isNotSet() {
			return actualValue.length() > 0;
		}

		@Override public int compareTo(Object stf) {
			return Integer.compare(((StringTrimFormat) stf).actualValue.length(), actualValue.length());
		}

		@Override public String getStringOfActualValue() {
			return actualValue;
		}

		@Override public boolean isEqual(ValueTrimFormat vtf) {
			if(vtf instanceof StringTrimFormat && this.actualValue
				.equalsIgnoreCase(((StringTrimFormat) vtf).actualValue))
				return true;
			else
				return false;
		}

		@Override public void setNoSet() {
			actualValue = "";
		}
	}

	// Convert all numeric values(i.e., double, float, int, long, ...) to number trim format
	public static class NumberTrimFormat extends ValueTrimFormat {
		private double actualValue; // save the actual value
		public char S; // signe of value "+" or "-"
		public char[] N; // array of none zero chars. Example: value = 0.00012345, N = [1,2,3,4,5]
		private String NString;

		public NumberTrimFormat(int colIndex, double actualValue) {
			this(actualValue);
			this.colIndex = colIndex;
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
			NString = toString();
		}

		public String getNString() {
			return NString;
		}

		@Override public String toString() {
			StringBuilder s = new StringBuilder();
			for(Character c : N)
				s.append(c);
			return s.toString();
		}

		@Override public boolean isNotSet() {
			return actualValue == 0;
		}

		@Override public ValueTrimFormat getACopy() {
			NumberTrimFormat copy = new NumberTrimFormat(colIndex, actualValue);
			copy.S = S;
			copy.N = N;
			return copy;
		}

		@Override public int compareTo(Object ntf) {
			int vc = Double.compare(((NumberTrimFormat) ntf).actualValue, actualValue);
			int sThis = actualValue >= 0 ? 0 : 1;
			int sThat = ((NumberTrimFormat) ntf).actualValue >= 0 ? 0 : 1;
			int nc = Integer.compare(((NumberTrimFormat) ntf).N.length + sThat, N.length + sThis);
			if(vc == 0)
				return 0;
			if(nc == 0)
				return vc;

			else
				return nc;
		}

		public double getActualValue() {
			return actualValue;
		}

		@Override public String getStringOfActualValue() {
			return actualValue + "";
		}

		@Override public boolean isEqual(ValueTrimFormat vtf) {
			if(vtf instanceof NumberTrimFormat && this.actualValue == ((NumberTrimFormat) vtf).actualValue)
				return true;
			else
				return false;
		}

		@Override public void setNoSet() {
			actualValue = 0;
			S = '+';
			N = new char[1];
			N[0] = '0';
			NString = null;

		}
	}

	// Convert Bool values(i.e., True/False, 0/1, T,F) to bool trim format
	public static class BooleanTrimFormat extends ValueTrimFormat {
		private boolean actualValue; // save the actual value

		public BooleanTrimFormat(int colIndex, boolean actualValue) {
			this(actualValue);
			this.colIndex = colIndex;
		}

		public BooleanTrimFormat(boolean actualValue) {
			this.actualValue = actualValue;
		}

		@Override public BooleanTrimFormat getACopy() {
			BooleanTrimFormat copy = new BooleanTrimFormat(colIndex, actualValue);
			return copy;
		}

		@Override public boolean isNotSet() {
			return false;
		}

		@Override public int compareTo(Object stf) {
			return 0;
		}

		@Override public String getStringOfActualValue() {
			return actualValue + "";
		}

		@Override public boolean isEqual(ValueTrimFormat vtf) {
			if(vtf instanceof BooleanTrimFormat && this.actualValue == ((BooleanTrimFormat) vtf).actualValue)
				return true;
			else
				return false;
		}

		@Override public void setNoSet() {}
	}
}
