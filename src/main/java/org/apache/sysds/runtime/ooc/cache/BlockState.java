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

package org.apache.sysds.runtime.ooc.cache;

public enum BlockState {
	HOT,
	WARM,
	EVICTING,
	READING,
	//DEFERRED_READ, // Deferred read
	COLD,
	REMOVED; // Removed state means that it is not owned by the cache anymore. It doesn't mean the object is dereferenced

	public boolean isAvailable() {
		return this == HOT || this == WARM || this == EVICTING || this == REMOVED;
	}

	public boolean isUnavailable() {
		return this == COLD || this == READING;
	}

	public boolean readScheduled() {
		return this == READING;
	}

	public boolean isBackedByDisk() {
		return switch(this) {
			case WARM, COLD, READING -> true;
			default -> false;
		};
	}
}
