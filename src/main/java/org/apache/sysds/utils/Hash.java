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

import java.util.Random;

import org.apache.commons.lang.NotImplementedException;

/**
 * A class containing different hashing functions.
 */
public class Hash {

	public enum HashType {
		StandardJava, LinearHash, ExpHash
	}

	static public int hash(Object o, HashType ht) {
		int hashcode = o.hashCode();
		if(hashcode == 0){
			hashcode = Integer.hashCode(13241);
		}
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

	// 32 random int values.
	// generated values:
	public static void main(String[] args) {
		Random r = new Random(1324121);
		for(int x = 0; x < 32; x++) {
			System.out.println(String.format("0x%08X,", r.nextInt()));
		}
	}

	private static int[] a = {0x21ae4036, 0x32435171, 0xac3338cf, 0xea97b40c, 0x0e504b22, 0x9ff9a4ef, 0x111d014d,
		0x934f3787, 0x6cd079bf, 0x69db5c31, 0xdf3c28ed, 0x40daf2ad, 0x82a5891c, 0x4659c7b0, 0x73dc0ca8, 0xdad3aca2,
		0x00c74c7e, 0x9a2521e2, 0xf38eb6aa, 0x64711ab6, 0x5823150a, 0xd13a3a9a, 0x30a5aa04, 0x0fb9a1da, 0xef785119,
		0xc9f0b067, 0x1e7dde42, 0xdda4a7b2, 0x1a1c2640, 0x297c0633, 0x744edb48, 0x19adce93};

	/**
	 * Compute linear hash function (32-bit signed integers to 32-bit signed integers) on list if integers provided in
	 * stdin, one integer per line
	 */
	public static int linearHash(int v) {
		return linearHash(v,a.length);
	}

	public static int linearHash(int v, int bits) {
		int res = 0;
		for(int i = 0; i < bits; i++) {
			res = (res << 1) + (Long.bitCount(a[i] & v) & 1);
		}
		return res;
	}


	/**
	 * Compute exponentially distributed hash values (in range 0..32)
	 * 
	 * (should cast result to a byte)
	 */
	public static int expHash(int x) {
		for(int value = 0; value < a.length; value++) {
			int dot = Long.bitCount(a[value] & x) & 1;
			if(dot != 0)
				return value + 1;
		}
		return a.length;
	}
}