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
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;

public class RawIndex {
	private final String raw;
	private final int rawLength;
	private final BitSet numberBitSet;
	private final BitSet dotBitSet;
	private final BitSet eBitSet;
	private final BitSet plusMinusBitSet;
	private BitSet reservedPositions;
	private BitSet backupReservedPositions;
	private HashMap<Double, ArrayList<Pair<Integer, Integer>>> actualNumericValues;
	private HashMap<Double, ArrayList<Pair<Integer, Integer>>> dotActualNumericValues;
	private HashMap<Double, ArrayList<Pair<Integer, Integer>>> dotEActualNumericValues;


	public RawIndex(String raw) {
		this.raw = raw;
		this.rawLength = raw.length();
		this.numberBitSet = new BitSet(rawLength);
		this.dotBitSet = new BitSet(rawLength);
		this.eBitSet = new BitSet(rawLength);
		this.plusMinusBitSet = new BitSet(rawLength);
		this.reservedPositions = new BitSet(rawLength);
		this.backupReservedPositions = new BitSet(rawLength);
		this.actualNumericValues = null;
		this.dotActualNumericValues = null;
		this.dotEActualNumericValues = new HashMap<>();

		for(int i = 0; i < this.rawLength; i++) {
			switch(raw.charAt(i)) {
				case '0':
				case '1':
				case '2':
				case '3':
				case '4':
				case '5':
				case '6':
				case '7':
				case '8':
				case '9':
					this.numberBitSet.set(i);
					break;
				case '+':
				case '-':
					this.plusMinusBitSet.set(i);
					break;
				case '.':
					this.dotBitSet.set(i);
					break;
				case 'e':
				case 'E':
					this.eBitSet.set(i);
					break;
			}
		}
		// Clean unnecessary sets
		// Clean for "."
		for(int i = dotBitSet.nextSetBit(0); i != -1; i = dotBitSet.nextSetBit(i + 1)) {
			boolean flag = false;
			if(i > 0) {
				if(i< rawLength -2) {
					flag = !numberBitSet.get(i - 1) &&
							!numberBitSet.get(i + 1) &&
							!plusMinusBitSet.get(i + 1) &&
							!eBitSet.get(i + 1);
					}
				}
				else if( i== rawLength-1){
					flag = !numberBitSet.get(i - 1);
				}
			else if(i==0){
				if(i < rawLength-2){
					flag = !numberBitSet.get(i + 1) &&
						!plusMinusBitSet.get(i + 1) &&
						!eBitSet.get(i + 1);
				}
				else if( i== rawLength-1){
					flag = true;
				}
			}

			if(flag)
				dotBitSet.set(i, false);
		}

		// Clean for "+/-"
		for(int i = plusMinusBitSet.nextSetBit(0); i != -1; i = plusMinusBitSet.nextSetBit(i + 1)) {
			boolean flag;
			if(i<rawLength-1){
				flag = numberBitSet.get(i + 1);
				if(!flag && i<rawLength-2)
					flag = dotBitSet.get(i+1) && numberBitSet.get(i+2);
			}
			else {
				flag = false;
			}
			if(!flag)
				plusMinusBitSet.set(i, false);
		}

		// Clean for "e/E"
		for(int i = eBitSet.nextSetBit(0); i != -1; i = eBitSet.nextSetBit(i + 1)) {
			boolean flag = false;
			if((i == 1 && !numberBitSet.get(0)) || i==0 || i==rawLength-1){
				flag = false;
			}
			else if(i>1 && i<rawLength-2){
				flag = numberBitSet.get(i-1) || (numberBitSet.get(i-2) && dotBitSet.get(i-1));
				if(flag)
					flag = numberBitSet.get(i+1) || (numberBitSet.get(i+2) && plusMinusBitSet.get(i+1));
			}
			else if(i==rawLength-2){
				flag = numberBitSet.get(rawLength-1);
			}
			if(!flag)
				eBitSet.set(i, false);
		}
		extractNumericDotEActualValues();
	}

	public Pair<Integer, Integer> findValue(Object value, Types.ValueType valueType){
		if(valueType.isNumeric())
			return findValue(UtilFunctions.getDouble(value));
		else if(valueType == Types.ValueType.STRING)
			return findValue(UtilFunctions.objectToString(value));
//		else if(valueType == Types.ValueType.BOOLEAN)
//			return findValue(UtilFunctions.objectToString())
		else
			return null;
	}

	public Pair<Integer, Integer> findValue(double value){
//		extractNumericActualValues();
//		if(actualNumericValues.containsKey(value)){
//			return getValuePositionAndLength(actualNumericValues.get(value));
//		}
//
//		extractNumericDotActualValues();
//		if(dotActualNumericValues.containsKey(value)){
//			return getValuePositionAndLength(dotActualNumericValues.get(value));
//		}
//
//		extractNumericDotEActualValues();
		if(dotEActualNumericValues.containsKey(value)){
			return getValuePositionAndLength(dotEActualNumericValues.get(value));
		}
		return null;
	}

	private Pair<Integer, Integer> findValue(String value){
		int index = this.raw.indexOf(value);
		if(index == -1)
			return null;
		else {
			for(int i= index; i<index+value.length();i++)
				if(reservedPositions.get(i))
					return null;
			reservedPositions.set(index, index+value.length());
			return new Pair<>(index, value.length());
		}
	}

	private Pair<Integer, Integer> getValuePositionAndLength(ArrayList<Pair<Integer, Integer>> list){
		for(Pair<Integer, Integer> p: list){
			if(!reservedPositions.get(p.getKey())) {
				reservedPositions.set(p.getKey(), p.getKey()+p.getValue());
				return p;
			}
		}
		return null;
	}

	private void extractNumericActualValues(){
		if(this.actualNumericValues == null)
			this.actualNumericValues = new HashMap<>();
		else
			return;
		StringBuilder sb = new StringBuilder();
		BitSet nBitSet = (BitSet) numberBitSet.clone();
		nBitSet.or(plusMinusBitSet);
		int pi = nBitSet.nextSetBit(0);
		sb.append(raw.charAt(pi));

		for(int i = nBitSet.nextSetBit(pi+1); i != -1; i = nBitSet.nextSetBit(i + 1)) {
			if(pi+sb.length() != i) {
				addActualValueToList(sb.toString(), pi, actualNumericValues);
				sb = new StringBuilder();
				sb.append(raw.charAt(i));
				pi = i;
			}
			else
				sb.append(raw.charAt(i));
		}
		if(sb.length()>0)
			addActualValueToList(sb.toString(), pi, actualNumericValues);
	}

	private void extractNumericDotActualValues(){
		if(this.dotActualNumericValues == null)
			this.dotActualNumericValues = new HashMap<>();
		else
			return;

		BitSet numericDotBitSet = (BitSet) numberBitSet.clone();
		numericDotBitSet.or(dotBitSet);
		numericDotBitSet.or(plusMinusBitSet);
		StringBuilder sb = new StringBuilder();
		int pi = numericDotBitSet.nextSetBit(0);
		sb.append(raw.charAt(pi));

		for(int i = numericDotBitSet.nextSetBit(pi+1); i != -1; i = numericDotBitSet.nextSetBit(i + 1)) {
			if(pi+sb.length() != i) {
				addActualValueToList(sb.toString(), pi, dotActualNumericValues);
				sb = new StringBuilder();
				sb.append(raw.charAt(i));
				pi = i;
			}
			else
				sb.append(raw.charAt(i));
		}
		if(sb.length()>0)
			addActualValueToList(sb.toString(), pi, dotActualNumericValues);
	}

	private void extractNumericDotEActualValues(){
//		if(this.dotEActualNumericValues == null)
//			this.dotEActualNumericValues = new HashMap<>();
//		else
//			return;

		BitSet numericDotEBitSet = (BitSet) numberBitSet.clone();
		numericDotEBitSet.or(dotBitSet);
		numericDotEBitSet.or(eBitSet);
		numericDotEBitSet.or(plusMinusBitSet);

		StringBuilder sb = new StringBuilder();
		int pi = numericDotEBitSet.nextSetBit(0);
		sb.append(raw.charAt(pi));

		for(int i = numericDotEBitSet.nextSetBit(pi+1); i != -1; i = numericDotEBitSet.nextSetBit(i + 1)) {
			if(pi+sb.length() != i) {
				addActualValueToList(sb.toString(), pi, dotEActualNumericValues);
				sb = new StringBuilder();
				sb.append(raw.charAt(i));
				pi = i;
			}
			else
				sb.append(raw.charAt(i));
		}
		if(sb.length()>0)
			addActualValueToList(sb.toString(), pi, dotEActualNumericValues);
	}

	private void addActualValueToList(String stringValue, Integer position, HashMap<Double, ArrayList<Pair<Integer, Integer>>> list){
		try {
			double d = UtilFunctions.getDouble(stringValue);
			Pair<Integer, Integer> pair = new Pair<Integer, Integer>(position, stringValue.length());
			if(!list.containsKey(d)) {
				ArrayList<Pair<Integer, Integer>> tmpList = new ArrayList<>();
				tmpList.add(pair);
				list.put(d, tmpList);
			}
			else
				list.get(d).add(pair);
		}
		catch(Exception e){

		}
	}

	public void printBitSets() {
		//		String numberBitSetStrng;
				String dotBitSetString="";
				String eBitSetString="";
				String plusMinusBitSetString="";
		//		String minusBitSetStrng;
		//		for(int i=0; i<rawLength;i++){
		//			numberBitSet.toString();
		//		}
		for(int i=0; i<rawLength;i++){
			if(dotBitSet.get(i))
				dotBitSetString+="1";
			else
				dotBitSetString+="0";

			if(eBitSet.get(i))
				eBitSetString+="1";
			else
				eBitSetString+="0";

			if(plusMinusBitSet.get(i))
				plusMinusBitSetString+="1";
			else
				plusMinusBitSetString+="0";
		}
		System.out.println("DOT="+dotBitSetString);
		System.out.println("E  ="+eBitSetString);
		System.out.println("+- ="+plusMinusBitSetString);

	}

	public void cloneReservedPositions(){
		this.backupReservedPositions = (BitSet) this.reservedPositions.clone();
	}

	public void restoreReservedPositions(){
		this.reservedPositions = this.backupReservedPositions;
	}

	public String getSubString(int start, int end){
		return raw.substring(start, end);
	}

	public int getRawLength() {
		return rawLength;
	}

	public String getRaw() {
		return raw;
	}
}
