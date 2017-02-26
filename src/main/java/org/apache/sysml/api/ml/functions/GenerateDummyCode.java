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

package org.apache.sysml.api.ml.functions;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.function.Function;

public class GenerateDummyCode implements Function<List<Object>, List<Object>> {
	
	private static final long serialVersionUID = 8288904231567560245L;

	@Override
	public List<Object> call(List<Object> arr) throws Exception {
		double value = (Double) arr.get(0);
		double minLabelValue = (Double) arr.get(2);
		double maxLabelValue = (Double) arr.get(3);
		List<Object> result = new ArrayList<Object>();
		List<Object> dummy = new ArrayList<Object>();
		result.add(arr.get(0));
		result.add(arr.get(1));
		
		for (int i = (int) minLabelValue; i <= (int) maxLabelValue; i++)
		{
			if (i == value)
				dummy.add(1.0);
			else
				dummy.add(0);
		}
		
		result.add(dummy);
		
		return result;
	}
}
