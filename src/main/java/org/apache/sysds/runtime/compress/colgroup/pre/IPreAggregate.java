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

import java.util.Arrays;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public interface IPreAggregate {
	public static final Log LOG = LogFactory.getLog(PreAggregateFactory.class.getName());

	public static ThreadLocal<int[]> aggPool = new ThreadLocal<int[]>() {
		@Override
		protected int[] initialValue() {
			return null;
		}
	};

	public static void setupThreadLocalMemory(int len) {
		if(aggPool.get() == null || aggPool.get().length < len) {
			int[] p = new int[len];
			aggPool.set(p);
		}
	}

	public static void cleanupThreadLocalMemory() {
		aggPool.remove();
	}

	public static int[] allocIVector(int len, boolean reset) {
		int[] p = aggPool.get();

		// if no pool
		if(p == null)
			return new int[len];

		// if to small
		if(p.length < len) {
			setupThreadLocalMemory(len);
			return aggPool.get();
		}

		if(reset)
			Arrays.fill(p, 0, len, 0);
		return p;
	}

	/**
	 * Increment index by 1.
	 * @param idx The index to increment.
	 */
	public void increment(int idx);

	/**
	 * Increment the index by v values.
	 * 
	 * @param idx The index to increment.
	 * @param v   The number of values to increment.
	 */
	public void increment(int idx, int v);
}
