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

package org.apache.sysds.test.component.ooc.cache;

import org.apache.sysds.runtime.ooc.cache.BlockEntry;
import org.apache.sysds.runtime.ooc.cache.BlockKey;
import org.apache.sysds.runtime.ooc.cache.BlockState;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;

final class BlockEntryTestAccess {
	private static final Constructor<BlockEntry> CTOR;
	private static final Method GET_DATA_UNSAFE;
	private static final Method SET_DATA_UNSAFE;
	private static final Method SET_STATE;

	static {
		try {
			CTOR = BlockEntry.class.getDeclaredConstructor(BlockKey.class, long.class, Object.class);
			CTOR.setAccessible(true);

			GET_DATA_UNSAFE = BlockEntry.class.getDeclaredMethod("getDataUnsafe");
			GET_DATA_UNSAFE.setAccessible(true);

			SET_DATA_UNSAFE = BlockEntry.class.getDeclaredMethod("setDataUnsafe", Object.class);
			SET_DATA_UNSAFE.setAccessible(true);

			SET_STATE = BlockEntry.class.getDeclaredMethod("setState", BlockState.class);
			SET_STATE.setAccessible(true);
		}
		catch(ReflectiveOperationException e) {
			throw new ExceptionInInitializerError(e);
		}
	}

	private BlockEntryTestAccess() {
		// Utility class
	}

	static BlockEntry newBlockEntry(BlockKey key, long size, Object data) {
		try {
			return CTOR.newInstance(key, size, data);
		}
		catch(ReflectiveOperationException e) {
			throw new RuntimeException("Failed to create BlockEntry via reflection", e);
		}
	}

	static Object getDataUnsafe(BlockEntry entry) {
		try {
			return GET_DATA_UNSAFE.invoke(entry);
		}
		catch(ReflectiveOperationException e) {
			throw new RuntimeException("Failed to call BlockEntry#getDataUnsafe via reflection", e);
		}
	}

	static void setDataUnsafe(BlockEntry entry, Object data) {
		try {
			SET_DATA_UNSAFE.invoke(entry, data);
		}
		catch(ReflectiveOperationException e) {
			throw new RuntimeException("Failed to call BlockEntry#setDataUnsafe via reflection", e);
		}
	}

	static void setState(BlockEntry entry, BlockState state) {
		try {
			SET_STATE.invoke(entry, state);
		}
		catch(ReflectiveOperationException e) {
			throw new RuntimeException("Failed to call BlockEntry#setState via reflection", e);
		}
	}
}
