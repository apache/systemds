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

package org.apache.sysds.runtime.compress.colgroup.pre;

import org.apache.sysds.runtime.compress.utils.IntIntMap;

public class MapPreAggregate implements IPreAggregate {

	private final IntIntMap map;

	protected MapPreAggregate(){
		map = new IntIntMap(64, 0.7f);
	}

	protected MapPreAggregate(int size, float fill){
		map = new IntIntMap(4096, 0.7f);
	}

	public int getSize() {
		return map.getCapacity() * 2;
	}

	public int getMapFreeValue() {
		return map.getFreeValue();
	}

	public void increment(int idx){
		map.inc(idx);
	}

	public void increment(int idx, int v){
		// if(v <= 0)
		//     throw new DMLCompressionException("Invalid increment " + v);
		map.inc(idx, v);
	}

	@Override
	public String toString() {
		return map.toString();
	}

	public int[] getMap(){
		return map.getMap();
	}
}
