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

package org.apache.sysds.runtime.util;

import org.apache.logging.log4j.util.TriConsumer;
import org.apache.sysds.runtime.DMLRuntimeException;

import java.util.HashMap;
import java.util.Map;

public class OOCJoin<T, O> {
	private Map<T, O> left;
	private Map<T, O> right;
	private TriConsumer<T, O, O> emitter;

	public OOCJoin(TriConsumer<T, O, O> emitter) {
		this.left = new HashMap<>();
		this.right = new HashMap<>();
		this.emitter = emitter;
	}

	public void addLeft(T idx, O item) {
		add(true, idx, item);
	}

	public void addRight(T idx, O item) {
		add(false, idx, item);
	}

	public void close() {
		synchronized (this) {
			if (!left.isEmpty() || !right.isEmpty())
				throw new DMLRuntimeException("There are still unprocessed items in the OOC join");
		}
	}

	public void add(boolean isLeft, T idx, O val) {
		Map<T, O> lookup = isLeft ? right : left;
		Map<T, O> store = isLeft ? left : right;
		O val2;

		synchronized (this) {
			val2 = lookup.remove(idx);

			if (val2 == null)
				store.put(idx, val);
		}

		if (val2 != null)
			emitter.accept(idx, isLeft ? val : val2, isLeft ? val2 : val);
	}
}
