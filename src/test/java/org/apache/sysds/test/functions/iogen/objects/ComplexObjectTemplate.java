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

import org.apache.sysds.common.Types;

import java.util.ArrayList;

public abstract class ComplexObjectTemplate extends JSONObjectTemplate {

	// Primitive Items
	protected Integer item1 = getRandomIntegerValue(false);
	protected Long item2 = getRandomLongValue(false);
	protected Float item3 = getRandomFloatValue(false);
	protected Double item4 = getRandomDoubleValue(false);
	protected String item5 = getRandomStringValue(false);
	protected Boolean item6 = getRandomBooleanValue(false);

	// Primitive List Items
	protected ArrayList<Integer> list1 = getRandomIntegerList();
	protected ArrayList<Long> list2 = getRandomLongList();
	protected ArrayList<Float> list3 = getRandomFloatList();
	protected ArrayList<Double> list4 = getRandomDoubleList();
	protected ArrayList<String> list5 = getRandomStringList();
	protected ArrayList<Boolean> list6 = getRandomBooleanList();

	// Object list Items
	private ArrayList<ComplexObject5> objectList5;

	public ComplexObjectTemplate() {
		if(!(this instanceof ComplexObject5)) {
			objectList5 = new ArrayList<>();
			int l = getRandomArrayLength();
			if(l == 0)
				objectList5 = null;
			else {
				for(int i = 0; i < l; i++)
					objectList5.add(new ComplexObject5());
			}
		}
	}

	protected ArrayList<Object> getJSONFlatValues() {
		ArrayList<Object> values = new ArrayList<>();
		values.add(item1);
		values.add(item2);
		values.add(item3);
		values.add(item4);
		values.add(item5);
		values.add(item6);
		values.addAll(getIntegerListValues(list1));
		values.addAll(getLongListValues(list2));
		values.addAll(getFloatListValues(list3));
		values.addAll(getDoubleListValues(list4));
		values.addAll(getStringListValues(list5));
		values.addAll(getBooleanListValues(list6));
		if(!(this instanceof ComplexObject5)) {
			if(objectList5 != null) {
				for(ComplexObject5 jno : objectList5)
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

	public ArrayList<Types.ValueType> getSchema() {
		ArrayList<Types.ValueType> schema = new ArrayList<>();
		schema.add(Types.ValueType.INT32);
		schema.add(Types.ValueType.INT64);
		schema.add(Types.ValueType.FP32);
		schema.add(Types.ValueType.FP64);
		schema.add(Types.ValueType.STRING);
		schema.add(Types.ValueType.BOOLEAN);

		for(int i = 0; i < maxLength; i++) {
			schema.add(Types.ValueType.INT32);
		}
		for(int i = 0; i < maxLength; i++) {
			schema.add(Types.ValueType.INT64);
		}
		for(int i = 0; i < maxLength; i++) {
			schema.add(Types.ValueType.FP32);
		}
		for(int i = 0; i < maxLength; i++) {
			schema.add(Types.ValueType.FP64);
		}
		for(int i = 0; i < maxLength; i++) {
			schema.add(Types.ValueType.STRING);
		}
		for(int i = 0; i < maxLength; i++) {
			schema.add(Types.ValueType.BOOLEAN);
		}

		if(!(this instanceof ComplexObject5)) {
			ComplexObject5 c5 = new ComplexObject5();
			for(int i = 0; i < maxLength; i++) {
				schema.addAll(c5.getSchema());
			}
		}
		return schema;

	}

	private static ArrayList<Object> getEmptyFlatObject() {
		ArrayList<Object> result = new ArrayList<>();
		int size = 6 + 6 * 10;
		for(int i = 0; i < size; i++)
			result.add(null);
		return result;
	}

	protected static ArrayList<Object> getEmptyFlatObject(Object o) {
		ArrayList<Object> result = getEmptyFlatObject();
		for(int i = 0; i < 10; i++)
			result.addAll(getEmptyFlatObject());

		if(o instanceof ComplexObject1)
			result.addAll(ComplexObject2.getEmptyFlatObject(o));

		if(o instanceof ComplexObject2)
			result.addAll(ComplexObject3.getEmptyFlatObject(o));

		if(o instanceof ComplexObject3)
			result.addAll(ComplexObject4.getEmptyFlatObject(o));

		if(o instanceof ComplexObject4)
			result.addAll(ComplexObject5.getEmptyFlatObject(o));
		return result;
	}

}
