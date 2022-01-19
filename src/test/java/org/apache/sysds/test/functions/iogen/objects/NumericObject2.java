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

public class NumericObject2 extends NumericObjectTemplate{

	// Object Items
	private NumericObject3 numericObject3 = new NumericObject3();

	public NumericObject2() {
		super();
	}

	@Override
	public ArrayList<Object> getJSONFlatValues() {
		ArrayList<Object> values = super.getJSONFlatValues();

		if(numericObject3 != null) {
			values.addAll(numericObject3.getJSONFlatValues());
		}
		else
			values.addAll(getEmptyFlatObject(this));

		return values;
	}
}
