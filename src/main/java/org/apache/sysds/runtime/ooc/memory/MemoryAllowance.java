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

package org.apache.sysds.runtime.ooc.memory;

public interface MemoryAllowance {
	boolean tryReserve(long bytes);
	void reserveBlocking(long bytes);
	void release(long bytes);
	long getUsedMemory();
	long getGrantedMemory();
	long getTargetMemory();
	void setTargetMemory(long targetMemory);
	void shutdown();
	boolean isShutdown();

	default void destroy() {
		shutdown();
	}

	default long getFreeMemory() {
		return Math.max(0, getGrantedMemory() - getUsedMemory());
	}

	default boolean isUnderPressure() {
		return getGrantedMemory() > getTargetMemory();
	}
}
