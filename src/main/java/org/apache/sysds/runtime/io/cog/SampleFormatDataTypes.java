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

package org.apache.sysds.runtime.io.cog;

/**
 * Enum for mapping sample formats of TIFF image data to names
 */
public enum SampleFormatDataTypes {
	UNSIGNED_INTEGER(1),
	SIGNED_INTEGER(2),
	FLOATING_POINT(3),
	UNDEFINED(4);

	private final int value;

	SampleFormatDataTypes(int value) {
		this.value = value;
	}

	public int getValue() {
		return value;
	}

	public static SampleFormatDataTypes valueOf(int value) {
		for (SampleFormatDataTypes dataType : SampleFormatDataTypes.values()) {
			if (dataType.getValue() == value) {
				return dataType;
			}
		}
		return null;
	}
}
