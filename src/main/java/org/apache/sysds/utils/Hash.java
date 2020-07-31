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

import org.apache.commons.lang3.NotImplementedException;

/**
 * A class containing different hashing functions.
 */
public class Hash {

	/**
	 * Available Hashing techniques
	 */
	public enum HashType {
		StandardJava, LinearHash, ExpHash
	}

	/**
	 * A random Array (except first value) used for Linear and Exp hashing, to integer domain.
	 */
	private static final int[] a = {0xFFFFFFFF, 0xB7825CBC, 0x10FA23F2, 0xD54E1532, 0x7590E53C, 0xECE6F631, 0x8954BF60,
		0x5BE38B88, 0xCA1D3AC0, 0xB2726F8E, 0xBADE7E7A, 0xCACD1184, 0xFB32BDAD, 0x2936C9D7, 0xB5B88D37, 0xD272D353,
		0xE139A063, 0xDACF6B87, 0x3568D521, 0x75C619EA, 0x7C2B8CBD, 0x012C3C7F, 0x0A621C37, 0x77274A12, 0x731D379A,
		0xE45E0D3B, 0xEAB4AE13, 0x10C440C7, 0x50CF2899, 0xD865BD46, 0xAABDF34F, 0x218FA0C3,};

	/**
	 * Generic hashing of java objects, not ideal for specific values so use the specific methods for specific types.
	 * 
	 * To Use the locality sensitive techniques override the objects hashcode function.
	 * 
	 * @param o  The Object to hash.
	 * @param ht The HashType to use.
	 * @return An int Hash value.
	 */
	public static int hash(Object o, HashType ht) {
		int hashcode = o.hashCode();
		switch(ht) {
			case StandardJava:
				return hashcode;
			case LinearHash:
				return linearHash(hashcode);
			case ExpHash:
				return expHash(hashcode);
			default:
				throw new NotImplementedException("Not Implemented hashing combination");
		}
	}

	/**
	 * Hash functions for double values.
	 * 
	 * @param o  The double value.
	 * @param ht The hashing function to apply.
	 * @return An int Hash value.
	 */
	public static int hash(double o, HashType ht) {
		switch(ht) {
			case StandardJava:
				// Here just for reference
				return new Double(o).hashCode();
			case LinearHash:
				// Altho Linear Hashing is locality sensitive, it is not in this case
				// since the bit positions for the double value is split in exponent and mantissa.
				// If the locality sensitive aspect is required use linear hash on an double value rounded to integer.
				long v = Double.doubleToLongBits(o);
				return linearHash((int) (v ^ (v >>> 32)));
			default:
				throw new NotImplementedException("Not Implemented hashing combination for double value");
		}
	}

	/**
	 * Compute the Linear hash of an int input value.
	 * 
	 * @param v The value to hash.
	 * @return The int hash.
	 */
	public static int linearHash(int v) {
		return linearHash(v, a.length);
	}

	/**
	 * Compute the Linear hash of an int input value, but only use the first bits of the linear hash.
	 * 
	 * @param v    The value to hash.
	 * @param bits The number of bits to use. up to maximum of 32.
	 * @return The hashed value
	 */
	public static int linearHash(int v, int bits) {
		int res = 0;
		for(int i = 0; i < bits; i++) {
			res = (res << 1) + (Long.bitCount(a[i] & v) & 1);
		}
		return res;
	}

	/**
	 * Compute exponentially distributed hash values in range 0..a.length
	 * 
	 * eg: 50% == 0 , 25% == 1 12.5 % == 2 etc.
	 * 
	 * Useful because you can estimate size of a collection by only maintaining the highest value found. from this hash.
	 * 
	 * @param x value to hash
	 * @return a hash value byte (only in the range of 0 to a.length)
	 */
	public static byte expHash(int x) {
		for(int value = 0; value < a.length; value++) {
			int dot = Long.bitCount(a[value] & x) & 1;
			if(dot != 0)
				return (byte) (value + 1);
		}
		return (byte) a.length;
	}
}
