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

package org.apache.sysds.utils;

/**
 * Memory Estimates is a helper class containing static classes that estimate the memory requirements of different types
 * of objects in java.
 * All estimates are worst case JVM x86-64bit uncompressed object pointers.
 */
public class MemoryEstimates {
	public static long byteArrayCost(int length) {
		long size = 0;
		size += 8; // Byte array Reference
		size += 20; // Byte array Object header
		if(length <= 4) { // byte array fills out the first 4 bytes differently than the later bytes.
			size += 4;
		}
		else { // byte array pads to next 8 bytes after the first 4.
			size += length;
			int diff = (length - 4) % 8;
			if(diff > 0) {
				size += 8 - diff;
			}
		}
		return size;
	}

	public static long charArrayCost(int length) {
		long size = 0;
		size += 8; // char array Reference
		size += 20; // char array Object header
		if(length <= 2) { // char array fills out the first 2 chars differently than the later bytes.
			size += 4;
		}
		else {
			size += length * 2;// 2 bytes per char
			int diff = (length * 2 - 4) % 8;
			if(diff > 0) {
				size += 8 - diff; // next object alignment
			}
		}
		return size;
	}

	public static long intArrayCost(int length) {
		long size = 0;
		size += 8; // _ptr int[] reference
		size += 20; // int array Object header
		if(length <= 1) {
			size += 4;
		}
		else {
			size += length * 4; // offsets 4 bytes per int
			if(length % 2 == 0) {
				size += 4;
			}
		}
		return size;
	}

	public static long doubleArrayCost(long length) {
		long size = 0;
		size += 8; // _values double array reference
		size += 20; // double array object header
		size += 4; // padding inside double array object to align to 8 bytes.
		size += 8 * length; // Each double fills 8 Bytes
		return size;
	}
}