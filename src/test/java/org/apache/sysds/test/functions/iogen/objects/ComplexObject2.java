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

public class ComplexObject2 extends ComplexObjectTemplate {

	// Object Items
	private ComplexObject3 complexObject3 = new ComplexObject3();

	public ComplexObject2() {
		super();
	}

	@Override
	public ArrayList<Object> getJSONFlatValues() {
		ArrayList<Object> values = super.getJSONFlatValues();

		if(complexObject3 != null) {
			values.addAll(complexObject3.getJSONFlatValues());
		}
		else
			values.addAll(getEmptyFlatObject(this));

		return values;
	}


	@Override
	public ArrayList<Types.ValueType> getSchema() {
		ArrayList<Types.ValueType> result = super.getSchema();
		result.addAll(complexObject3.getSchema());
		return result;
	}
}
