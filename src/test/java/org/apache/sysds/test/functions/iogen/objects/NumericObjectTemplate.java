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

import java.util.ArrayList;

public abstract class NumericObjectTemplate extends JSONObjectTemplate {

	// Primitive Items
	protected Integer item1 = getRandomIntegerValue(false);
	protected Long item2 = getRandomLongValue(false);
	protected Float item3 = getRandomFloatValue(false);
	protected Double item4 = getRandomDoubleValue(false);

	// Primitive List Items
	protected ArrayList<Integer> list1 = getRandomIntegerList();
	protected ArrayList<Long> list2 = getRandomLongList();
	protected ArrayList<Float> list3 = getRandomFloatList();
	protected ArrayList<Double> list4 = getRandomDoubleList();

	// Object list Items
	private ArrayList<NumericObject5> objectList5;

	public NumericObjectTemplate() {
		if(!(this instanceof NumericObject5)) {
			objectList5 = new ArrayList<>();
			int l = getRandomArrayLength();
			if(l == 0)
				objectList5 = null;
			else {
				for(int i = 0; i < l; i++)
					objectList5.add(new NumericObject5());
			}
		}
	}

	protected ArrayList<Object> getJSONFlatValues() {
		ArrayList<Object> values = new ArrayList<>();
		values.add(item1);
		values.add(item2);
		values.add(item3);
		values.add(item4);
		values.addAll(getIntegerListValues(list1));
		values.addAll(getLongListValues(list2));
		values.addAll(getFloatListValues(list3));
		values.addAll(getDoubleListValues(list4));
		if(!(this instanceof NumericObject5)) {
			if(objectList5 != null) {
				for(NumericObject5 jno : objectList5)
					values.addAll(jno.getJSONFlatValues());
				for(int i = objectList5.size(); i < maxLength; i++)
					values.addAll(getEmptyFlatObject());
			}
			else
				for(int i = 0; i < maxLength; i++)
					values.addAll(getEmptyFlatObject());
		}

		return values;
	}

	private static ArrayList<Object> getEmptyFlatObject() {
		ArrayList<Object> result = new ArrayList<>();
		int size = 4 + 4 * 10;
		for(int i = 0; i < size; i++)
			result.add(null);
		return result;
	}

	protected static ArrayList<Object> getEmptyFlatObject(Object o) {
		ArrayList<Object> result = getEmptyFlatObject();
		for(int i = 0; i < 10; i++)
			result.addAll(getEmptyFlatObject());

		if(o instanceof NumericObject1)
			result.addAll(NumericObject2.getEmptyFlatObject(o));

		if(o instanceof NumericObject2)
			result.addAll(NumericObject3.getEmptyFlatObject(o));

		if(o instanceof NumericObject3)
			result.addAll(NumericObject4.getEmptyFlatObject(o));

		if(o instanceof NumericObject4)
			result.addAll(NumericObject5.getEmptyFlatObject(o));
		return result;
	}
}
