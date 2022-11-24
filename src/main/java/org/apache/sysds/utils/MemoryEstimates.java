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
 * of objects in java. All estimates are worst case JVM x86-64bit uncompressed object pointers.
 * 
 * This in practice means that the objects are most commonly smaller, for instance the object references are often time.
 * 
 * If the memory pressure is low (there is a low number of allocated objects) then object pointers are 4 bits.
 */
public class MemoryEstimates {

	// private static final Log LOG = LogFactory.getLog(MemoryEstimates.class.getName());

	/**
	 * Get the worst case memory usage of an java.util.BitSet java object.
	 * 
	 * @param length The length of the array.
	 * @return The memory estimate in bytes
	 */
	public static double bitSetCost(long length) {
		double size = 0;
		size += 8; // object reference
		size += longArrayCost(length / 64 + (length % 64 > 0 ? 1 : 0));
		size += 4; // words in Use
		size += 1; // size is Sticky
		size += 3; // padding.
		return size;
	}

	/**
	 * Get the worst case memory usage of an array of booleans.
	 * 
	 * Unfortunately in java booleans are allocated as bytes.
	 * 
	 * @param length The length of the array.
	 * @return The memory estimate in bytes.
	 */
	public static double booleanArrayCost(long length) {
		return byteArrayCost(length);
	}

	/**
	 * Get the worst case memory usage of an array of bytes.
	 * 
	 * @param length The length of the array.
	 * @return The memory estimate in bytes
	 */
	public static double byteArrayCost(long length) {
		long size = 0;
		size += 8; // Byte array Reference
		size += 20; // Byte array Object header
		if(length <= 4) { // byte array fills out the first 4 bytes differently than the later bytes.
			size += 4;
		}
		else { // byte array pads to next 8 bytes after the first 4.
			size += length;
			double diff = (length - 4) % 8;
			if(diff > 0) {
				size += 8 - diff;
			}
		}
		return size;
	}

	/**
	 * Get the worst case memory usage of an array of chars.
	 * 
	 * @param length The length of the array.
	 * @return The memory estimate in bytes
	 */
	public static double charArrayCost(long length) {
		double size = 0;
		size += 8; // char array Reference
		size += 20; // char array Object header
		if(length <= 2) { // char array fills out the first 2 chars differently than the later bytes.
			size += 4;
		}
		else {
			size += length * 2;// 2 bytes per char
			double diff = (length * 2 - 4) % 8;
			if(diff > 0) {
				size += 8 - diff; // next object alignment
			}
		}
		return size;
	}

	/**
	 * Get the worst case memory usage of an array of integers.
	 * 
	 * @param length The length of the array.
	 * @return The memory estimate in bytes
	 */
	public static double intArrayCost(long length) {
		double size = 0;
		size += 8; // _ptr int[] reference
		size += 20; // int array Object header
		if(length <= 1) {
			size += 4;
		}
		else {
			size += 4d * length; // offsets 4 bytes per int
			if(length % 2 == 0)
				size += 4;
		}
		return size;
	}

	/**
	 * Get the worst case memory usage of an array of integers.
	 * 
	 * @param length The length of the array.
	 * @return The memory estimate in bytes.
	 */
	public static double floatArrayCost(long length) {
		// exact same size as intArrayCost
		return intArrayCost(length);
	}

	/**
	 * Get the worst case memory usage of an array of doubles.
	 * 
	 * @param length The length of the array.
	 * @return The memory estimate in bytes
	 */
	public static final double doubleArrayCost(long length) {
		double size = 0;
		size += 8; // _values double array reference
		size += 20; // double array object header
		size += 4; // padding inside double array object to align to 8 bytes.
		size += 8d * length; // Each double fills 8 Bytes
		return size;
	}

	/**
	 * Get the worst case memory usage for an array of objects.
	 * 
	 * Note this does not take into account each objects allocated variables, just the objects themselves.
	 * 
	 * @param length The length of the array.
	 * @return The memory estimate in bytes
	 */
	public static final double objectArrayCost(long length) {
		double size = 0;
		size += 8d; // reference to array
		size += 20d; // header
		size += 4d; // padding before first reference
		size += 8d * length; // references to all objects.
		return size;
	}

	/**
	 * Get the worst case memory usage for an array of longs
	 * 
	 * @param length The length of the array.
	 * @return The memory estimate in bytes
	 */
	public static final double longArrayCost(long length) {
		// exactly the same size as a double array
		return doubleArrayCost(length);
	}

	/**
	 * Get the worst case memory usage of an array containing strings.
	 * 
	 * In case anyone is wondering ... strings are expensive.
	 * 
	 * @param strings The strings array.
	 * @return The array memory cost
	 */
	public static final double stringArrayCost(String[] strings) {
		double size = 0;
		for(int i = 0; i < strings.length; i++)
			size += stringCost(strings[i]);
		return size;
	}

	/**
	 * Get the worst case memory usage of a single string.
	 * 
	 * In case anyone is wondering ... strings are expensive.
	 * 
	 * @param value The String to measure
	 * @return The size in memory.
	 */
	public static double stringCost(String value) {
		return value == null ? 16 + 8 : stringCost(value.length()); // char array
	}

	/**
	 * Get the worst case memory usage of a single string, assuming given length.
	 * 
	 * @param length The length of the string
	 * @return The size in memory
	 */
	public static double stringCost(int length) {
		return 16 // object
			+ 4 // hash
			+ 8 // array ref
			+ 32 + length; // char array
	}

}
