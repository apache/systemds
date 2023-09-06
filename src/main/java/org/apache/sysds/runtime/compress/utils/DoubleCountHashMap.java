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

package org.apache.sysds.runtime.compress.utils;

import org.apache.sysds.runtime.compress.utils.ACount.DCounts;

public final class DoubleCountHashMap extends ACountHashMap<Double> {

	public DoubleCountHashMap() {
		super();
	}

	public DoubleCountHashMap(int init_capacity) {
		super(init_capacity);
	}

	protected DCounts[] create(int size) {
		return new DCounts[size];
	}

	protected int hash(Double key) {
		return DCounts.hashIndex(key);
	}

	protected final DCounts create(double key, int id) {
		return new DCounts(key, id);
	}

	protected final DCounts create(Double key, int id) {
		return new DCounts(key, id);
	}

	public double[] getDictionary() {
		double[] ret = new double[size];
		for(int i = 0; i < data.length; i++) {
			ACount<Double> e = data[i];
			while(e != null) {
				ret[e.id] = e.key();
				e = e.next();
			}
		}

		return ret;
	}

	public void replaceWithUIDsNoZero() {
		int i = 0;
		Double z = Double.valueOf(0.0);
		for(ACount<Double> e : data) {
			while(e != null) {
				if(!e.key().equals(z))
					e.id = i++;
				else
					e.id = -1;
				e = e.next();
			}
		}

	}

	@Override
	public DoubleCountHashMap clone() {
		DoubleCountHashMap ret = new DoubleCountHashMap(size);
		for(ACount<Double> e : data)
			ret.appendValue(e);
		ret.size = size;
		return ret;
	}
}
