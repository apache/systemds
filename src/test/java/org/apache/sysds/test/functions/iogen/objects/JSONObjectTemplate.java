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

package org.apache.sysds.test.functions.iogen.objects;

import com.google.gson.Gson;

import java.util.ArrayList;
import java.util.Random;

public abstract class JSONObjectTemplate {

	private transient double min = 1;

	private transient double max = 100;

	private transient int minLength = 0;

	protected transient int maxLength = 10;

	protected String getRandomStringValue(boolean nullable) {
		String alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
		StringBuilder salt = new StringBuilder();
		Random rnd = new Random();
		int length = getRandomArrayLength();
		if(nullable && length == 0)
			return null;
		while(salt.length() < length) {
			int index = (int) (rnd.nextFloat() * alphabet.length());
			salt.append(alphabet.charAt(index));
		}
		return salt.toString();
	}

	protected Integer getRandomIntegerValue(boolean nullable) {
		Random rnd = new Random();
		int value = (int) (rnd.nextFloat() * (max - min) + min);

		// return some null values
		if(nullable && (value > max * 0.9 || value < min * 0.9))
			return null;
		else
			return value;
	}

	protected Float getRandomFloatValue(boolean nullable) {
		Random rnd = new Random();
		float value = (float) (rnd.nextFloat() * (max - min) + min);

		// return some null values
		if(nullable && (value > max * 0.9 || value < min * 0.9))
			return null;
		else
			return value;
	}

	protected Double getRandomDoubleValue(boolean nullable) {
		Random rnd = new Random();
		double value = rnd.nextDouble() * (max - min) + min;

		// return some null values
		if(nullable && (value > max * 0.9 || value < min * 0.9))
			return null;
		else
			return value;
	}

	protected Long getRandomLongValue(boolean nullable) {
		Random rnd = new Random();
		long value = (long) (rnd.nextDouble() * (max - min) + min);

		// return some null values
		if(nullable && (value > max * 0.9 || value < min * 0.9))
			return null;
		else
			return value;
	}

	protected Boolean getRandomBooleanValue(boolean nullable) {
		Random rnd = new Random();
		long value = (long) (rnd.nextFloat() * (max - min) + min);

		// return some null values
		if(nullable && (value > max * 0.9 || value < min * 0.9))
			return null;
		else
			return value > 0;
	}

	public String getJSON() {
		Gson gson = new Gson();
		return gson.toJson(this);
	}

	protected ArrayList<Integer> getRandomIntegerList() {
		int l = getRandomArrayLength();
		if(l == 0)
			return null;
		else {
			ArrayList<Integer> list = new ArrayList<>();
			for(int i = 0; i < l; i++)
				list.add(getRandomIntegerValue(false));
			return list;
		}
	}

	protected ArrayList<Long> getRandomLongList() {
		int l = getRandomArrayLength();
		if(l == 0)
			return null;
		else {
			ArrayList<Long> list = new ArrayList<>();
			for(int i = 0; i < l; i++)
				list.add(getRandomLongValue(false));
			return list;
		}
	}

	protected ArrayList<Float> getRandomFloatList() {
		int l = getRandomArrayLength();
		if(l == 0)
			return null;
		else {
			ArrayList<Float> list = new ArrayList<>();
			for(int i = 0; i < l; i++)
				list.add(getRandomFloatValue(false));
			return list;
		}
	}

	protected ArrayList<Double> getRandomDoubleList() {
		int l = getRandomArrayLength();
		if(l == 0)
			return null;
		else {
			ArrayList<Double> list = new ArrayList<>();
			for(int i = 0; i < l; i++)
				list.add(getRandomDoubleValue(false));
			return list;
		}
	}

	protected ArrayList<Boolean> getRandomBooleanList() {
		int l = getRandomArrayLength();
		if(l == 0)
			return null;
		else {
			ArrayList<Boolean> list = new ArrayList<>();
			for(int i = 0; i < l; i++)
				list.add(getRandomBooleanValue(false));
			return list;
		}
	}

	protected ArrayList<String> getRandomStringList() {
		int l = getRandomArrayLength();
		if(l == 0)
			return null;
		else {
			ArrayList<String> list = new ArrayList<>();
			for(int i = 0; i < l; i++)
				list.add(getRandomStringValue(false));
			return list;
		}
	}

	protected int getRandomArrayLength() {
		//Random rnd = new Random();
		//return (int) (rnd.nextFloat() * (maxLength - minLength) + minLength);
		return maxLength;
	}

	protected ArrayList<Object> getIntegerListValues(ArrayList<Integer> list) {
		ArrayList<Object> result = new ArrayList<>();
		int index = 0;
		if(list != null) {
			result.addAll(list);
			index = list.size();
		}
		Integer n = 0;
		for(int i = index; i < maxLength; i++)
			result.add(n);
		return result;
	}

	protected ArrayList<Object> getLongListValues(ArrayList<Long> list) {
		ArrayList<Object> result = new ArrayList<>();
		int index = 0;
		if(list != null) {
			result.addAll(list);
			index = list.size();
		}
		Long n = 0L;
		for(int i = index; i < maxLength; i++)
			result.add(n);
		return result;
	}

	protected ArrayList<Object> getFloatListValues(ArrayList<Float> list) {
		ArrayList<Object> result = new ArrayList<>();
		int index = 0;
		if(list != null) {
			result.addAll(list);
			index = list.size();
		}
		Float n = 0f;
		for(int i = index; i < maxLength; i++)
			result.add(n);
		return result;
	}

	protected ArrayList<Object> getDoubleListValues(ArrayList<Double> list) {
		ArrayList<Object> result = new ArrayList<>();
		int index = 0;
		if(list != null) {
			result.addAll(list);
			index = list.size();
		}
		Double n = 0d;
		for(int i = index; i < maxLength; i++)
			result.add(n);
		return result;
	}

	protected ArrayList<Object> getBooleanListValues(ArrayList<Boolean> list) {
		ArrayList<Object> result = new ArrayList<>();
		int index = 0;
		if(list != null) {
			result.addAll(list);
			index = list.size();
		}
		Boolean n = false;
		for(int i = index; i < maxLength; i++)
			result.add(n);
		return result;
	}

	protected ArrayList<Object> getStringListValues(ArrayList<String> list) {
		ArrayList<Object> result = new ArrayList<>();
		int index = 0;
		if(list != null) {
			result.addAll(list);
			index = list.size();
		}
		String n = "";
		for(int i = index; i < maxLength; i++)
			result.add(n);
		return result;
	}

}
