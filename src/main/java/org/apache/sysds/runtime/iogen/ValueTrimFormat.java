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
import org.apache.sysds.runtime.util.UtilFunctions;

import java.text.DecimalFormat;

public class ValueTrimFormat implements Comparable<ValueTrimFormat> {

	// save the col index of the value on the Matrix.
	// We need this value when we want to reorder matrix cols
	private final int colIndex;
	private Object actualValue;

	// Convert all numeric values(i.e., double, float, int, long, ...) to number trim format
	public char S; // signe of value "+" or "-"
	private char[] N; // array of none zero chars. Example: value = 0.00012345, N = [1,2,3,4,5]
	private String NString;
	private Types.ValueType valueType;

	public ValueTrimFormat(int actualValue) {
		this(-1, Types.ValueType.INT32, actualValue);
	}

	public ValueTrimFormat(String actualValue) {
		this.valueType = Types.ValueType.STRING;
		this.actualValue = actualValue;
		this.colIndex = -1;
	}

	public ValueTrimFormat(int colIndex, Types.ValueType vt, Object o) {
		this.valueType = vt;
		this.colIndex = colIndex;
		this.actualValue = o;
		if(vt.isNumeric()) {
			double value = UtilFunctions.getDouble(o);

			// remove scientific format
			DecimalFormat decimalFormat = new DecimalFormat("0.000000000000000000000000000000");
			String stringValue = decimalFormat.format(value);
			if(value == 0) {
				S = '+';
				N = new char[] {'0'};
			}
			else {
				S = (value < 0) ? '-' : '+';
				if((o instanceof Long || o instanceof Integer) && stringValue.contains(".")) {
					stringValue = stringValue.substring(0, stringValue.indexOf("."));
				}
				numberTrimFormat(stringValue);
			}
			StringBuilder s = new StringBuilder();
			for(Character c : N)
				s.append(c);
			NString = s.toString();
		}
		else if(vt != Types.ValueType.STRING && vt != Types.ValueType.BITSET) {
			throw new RuntimeException("Don't support  value type format!");
		}
	}

	private void numberTrimFormat(String stringValue) {
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

	public double getDoubleActualValue() {
		return UtilFunctions.getDouble(actualValue);
	}

	// Get a copy of value
	public ValueTrimFormat getACopy() {
		ValueTrimFormat copy = null;
		if(valueType.isNumeric()) {
			copy = new ValueTrimFormat(colIndex, valueType, getDoubleActualValue());
			copy.S = S;
			copy.N = N;
		}
		else {
			copy = new ValueTrimFormat(colIndex, valueType, actualValue);
		}
		return copy;
	}

	// Check the value is a not set value
	public boolean isNotSet() {

		if(this.valueType == Types.ValueType.STRING)
			return actualValue == null || ((String) actualValue).length() == 0;
		else if(this.valueType.isNumeric())
			return getDoubleActualValue() == 0;
		else if(this.valueType == Types.ValueType.BITSET)
			return actualValue == null || !((Boolean) actualValue);
		return true;
	}

	// Set as NoSet
	public void setNoSet() {
		if(this.valueType == Types.ValueType.STRING)
			actualValue = "";
		else if(this.valueType.isNumeric()) {
			actualValue = (double) 0;
			S = '+';
			N = new char[] {'0'};
			NString = null;
		}
		else if(this.valueType == Types.ValueType.BITSET)
			actualValue = null;
	}

	// Get String of actual value
	public String getStringOfActualValue() {
		return UtilFunctions.objectToString(actualValue);
	}

	public boolean isEqual(ValueTrimFormat vtf) {
		if(vtf.getValueType() != this.getValueType())
			return false;
		else if(vtf.getValueType() == Types.ValueType.FP32)
			return ((Float) this.actualValue).compareTo((Float) vtf.actualValue) == 0;
		return UtilFunctions.compareTo(valueType, this.actualValue, vtf.actualValue) == 0;
	}

	public int getColIndex() {
		return colIndex;
	}

	private static int getLength(ValueTrimFormat vtf) {
		Types.ValueType vt = vtf.valueType;
		int len = -1;
		if(vt == Types.ValueType.STRING )
			len = vtf.getStringOfActualValue().length();
		else if(vt == Types.ValueType.BITSET)
			len = 1;
		return len;
	}

	@Override
	public int compareTo(ValueTrimFormat vtf) {
		Types.ValueType vt = vtf.valueType;
		if(vt.isNumeric() && this.valueType.isNumeric()) {
			return compareNumericVTF(vtf, this);
		}
		else if(vt.isNumeric() && this.valueType == Types.ValueType.STRING) {
			return -1;
		}
		else if(vt == Types.ValueType.STRING && this.valueType.isNumeric()) {
			try {
				Double d = Double.parseDouble(vtf.getStringOfActualValue());
				ValueTrimFormat vtfs = new ValueTrimFormat(-1,Types.ValueType.FP64,d);
				return compareNumericVTF(vtfs, this);
			}
			catch(Exception exception){
				return 1;
			}
		}
		else
			return Integer.compare(getLength(vtf), getLength(this));
	}

	private static int compareNumericVTF(ValueTrimFormat vtf1, ValueTrimFormat vtf2){
		double dv1 = vtf1.getDoubleActualValue();
		double dv2 = vtf2.getDoubleActualValue();
		int vc = Double.compare(dv1, dv2);

		if(vc == 0)
			return 0;

		int s1 = dv1 >= 0 ? 0 : 1;
		int s2 = dv2 >= 0 ? 0 : 1;
		int nc = Integer.compare(vtf1.N.length + s1, vtf2.N.length + s2);
		if(nc == 0)
			return Double.compare(Math.abs(dv1), Math.abs(dv2));
		else
			return nc;
	}

	public String getNString() {
		return NString;
	}

	public Types.ValueType getValueType() {
		return valueType;
	}
}
