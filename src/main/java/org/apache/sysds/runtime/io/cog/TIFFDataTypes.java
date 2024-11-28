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
 * Enum for mapping data types of IFD tags in TIFF to readable names
 */
public enum TIFFDataTypes {
	BYTE(1),
	ASCII(2),
	SHORT(3),
	LONG(4),
	RATIONAL(5),
	SBYTE(6),
	UNDEFINED(7),
	SSHORT(8),
	SLONG(9),
	SRATIONAL(10),
	FLOAT(11),
	DOUBLE(12),
	LONG8(16),
	SLONG8(17),
	IFD8(18);

	private final int value;

	TIFFDataTypes(int value) {
		this.value = value;
	}

	public int getValue() {
		return value;
	}

	public int getSize() {
		switch(this) {
			case BYTE:
			case ASCII:
			case SBYTE:
			case UNDEFINED:
				return 1;
			case SHORT:
			case SSHORT:
				return 2;
			case LONG:
			case SLONG:
			case FLOAT:
				return 4;
			case RATIONAL:
			case SRATIONAL:
			case DOUBLE:
			case LONG8:
			case SLONG8:
			case IFD8:
				return 8;
			default:
				return 0;
		}
	}

	public static TIFFDataTypes valueOf(int value) {
		for (TIFFDataTypes dataType : TIFFDataTypes.values()) {
			if (dataType.getValue() == value) {
				return dataType;
			}
		}
		return null;
	}
}
