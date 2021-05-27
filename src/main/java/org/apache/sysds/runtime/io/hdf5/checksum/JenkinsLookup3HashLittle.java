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


package org.apache.sysds.runtime.io.hdf5.checksum;

import java.nio.ByteBuffer;

import static java.lang.Byte.toUnsignedInt;
import static java.lang.Integer.rotateLeft;

/**
 * Hash used for HDF5 consistency checking Java code inspired by the Bob Jenkins C
 * code.
 * <p>
 * lookup3.c, by Bob Jenkins, May 2006, Public Domain.
 *
 * You can use this free for any purpose.  It's in the public domain.
 * It has no warranty.
 * </p>
 *
 * @author James Mudd
 * @see <a href="http://burtleburtle.net/bob/c/lookup3.c">lookup3.c</a>
 */
public final class JenkinsLookup3HashLittle {

	private static final int INITIALISATION_CONSTANT = 0xDEADBEEF;

	private JenkinsLookup3HashLittle() {
		throw new AssertionError("No instances of JenkinsLookup3HashLittle");
	}

	/**
	 * Equivalent to {@link #hash(byte[] bytes, int initialValue)} with initialValue = 0
	 *
	 * @param key bytes to hash
	 * @return hash value
	 */
	public static int hash(final byte[] key) {
		return hash(key, 0);
	}

	/**
	 * Equivalent to {@link #hash(ByteBuffer byteBuffer, int initialValue)} with initialValue = 0
	 *
	 * @param byteBuffer bytes to hash
	 * @return hash value
	 */
	public static int hash(final ByteBuffer byteBuffer) {
		return hash(byteBuffer, 0);
	}

	/**
	 * <p>
	 * The best hash table sizes are powers of 2.  There is no need to do mod
	 * a prime (mod is sooo slow!).  If you need less than 32 bits, use a bitmask.
	 * For example, if you need only 10 bits, do
	 * <code>h = (h {@literal &} hashmask(10));</code>
	 * In which case, the hash table should have hashsize(10) elements.
	 * </p>
	 *
	 * <p>If you are hashing n strings byte[][] k, do it like this:
	 * for (int i = 0, h = 0; i {@literal <} n; ++i) h = hash(k[i], h);
	 * </p>
	 *
	 * <p>By Bob Jenkins, 2006.  bob_jenkins@burtleburtle.net.  You may use this
	 * code any way you wish, private, educational, or commercial.  It's free.
	 * </p>
	 *
	 * <p>
	 * Use for hash table lookup, or anything where one collision in 2^^32 is
	 * acceptable.  Do NOT use for cryptographic purposes.
	 * </p>
	 *
	 * @param bytes bytes to hash
	 * @param initialValue can be any integer value
	 * @return hash value.
	 */
	public static int hash(final byte[] bytes, final int initialValue) {
		return hash(ByteBuffer.wrap(bytes), initialValue);
	}

	/*
	 * Final mixing of 3 32-bit values (a,b,c) into c
	 *
	 * Pairs of (a,b,c) values differing in only a few bits will usually
	 * produce values of c that look totally different.  This was tested for
	 * - pairs that differed by one bit, by two bits, in any combination
	 *   of top bits of (a,b,c), or in any combination of bottom bits of
	 *   (a,b,c).
	 *
	 * - "differ" is defined as +, -, ^, or ~^.  For + and -, I transformed
	 *   the output delta to a Gray code (a^(a>>1)) so a string of 1's (as
	 *   is commonly produced by subtraction) look like a single 1-bit
	 *   difference.
	 *
	 * - the base values were pseudorandom, all zero but one bit set, or
	 *   all zero plus a counter that starts at zero.
	 *
	 * These constants passed:
	 *   14 11 25 16 4 14 24
	 *   12 14 25 16 4 14 24
	 * and these came close:
	 *    4  8 15 26 3 22 24
	 *   10  8 15 26 3 22 24
	 *   11  8 15 26 3 22 24
	 */
	private static int finalMix(int a, int b, int c) {
		c ^= b;
		c -= rotateLeft(b, 14);
		a ^= c;
		a -= rotateLeft(c, 11);
		b ^= a;
		b -= rotateLeft(a, 25);
		c ^= b;
		c -= rotateLeft(b, 16);
		a ^= c;
		a -= rotateLeft(c, 4);
		b ^= a;
		b -= rotateLeft(a, 14);
		c ^= b;
		c -= rotateLeft(b, 24);

		return c;
	}

	/**
	 * <p>
	 * The best hash table sizes are powers of 2.  There is no need to do mod
	 * a prime (mod is sooo slow!).  If you need less than 32 bits, use a bitmask.
	 * For example, if you need only 10 bits, do
	 * <code>h = (h {@literal &} hashmask(10));</code>
	 * In which case, the hash table should have hashsize(10) elements.
	 * </p>
	 *
	 * <p>If you are hashing n strings byte[][] k, do it like this:
	 * for (int i = 0, h = 0; i {@literal <} n; ++i) h = hash(k[i], h);
	 * </p>
	 *
	 * <p>By Bob Jenkins, 2006.  bob_jenkins@burtleburtle.net.  You may use this
	 * code any way you wish, private, educational, or commercial.  It's free.
	 * </p>
	 *
	 * <p>
	 * Use for hash table lookup, or anything where one collision in 2^^32 is
	 * acceptable.  Do NOT use for cryptographic purposes.
	 * </p>
	 *
	 * @param byteBuffer to hash
	 * @param initialValue can be any integer value
	 * @return hash value.
	 */
	public static int hash(final ByteBuffer byteBuffer, final int initialValue) {

		// Initialise a, b and c
		int a = INITIALISATION_CONSTANT + byteBuffer.remaining() + initialValue;
		int b = a;
		int c = b;

		while (byteBuffer.remaining() > 12) {
			a += toUnsignedInt(byteBuffer.get());
			a += toUnsignedInt(byteBuffer.get()) << 8;
			a += toUnsignedInt(byteBuffer.get()) << 16;
			a += toUnsignedInt(byteBuffer.get()) << 24;
			b += toUnsignedInt(byteBuffer.get());
			b += toUnsignedInt(byteBuffer.get()) << 8;
			b += toUnsignedInt(byteBuffer.get()) << 16;
			b += toUnsignedInt(byteBuffer.get()) << 24;
			c += toUnsignedInt(byteBuffer.get());
			c += toUnsignedInt(byteBuffer.get()) << 8;
			c += toUnsignedInt(byteBuffer.get()) << 16;
			c += toUnsignedInt(byteBuffer.get()) << 24;

			/*
			 * mix -- mix 3 32-bit values reversibly.
			 * This is reversible, so any information in (a,b,c) before mix() is
			 * still in (a,b,c) after mix().
			 *
			 * If four pairs of (a,b,c) inputs are run through mix(), or through
			 * mix() in reverse, there are at least 32 bits of the output that
			 * are sometimes the same for one pair and different for another pair.
			 *
			 * This was tested for:
			 * - pairs that differed by one bit, by two bits, in any combination
			 *   of top bits of (a,b,c), or in any combination of bottom bits of
			 *   (a,b,c).
			 * - "differ" is defined as +, -, ^, or ~^.  For + and -, I transformed
			 *   the output delta to a Gray code (a^(a>>1)) so a string of 1's (as
			 *    is commonly produced by subtraction) look like a single 1-bit
			 *    difference.
			 * - the base values were pseudorandom, all zero but one bit set, or
			 *   all zero plus a counter that starts at zero.
			 *
			 * Some k values for my "a-=c; a^=rot(c,k); c+=b;" arrangement that
			 * satisfy this are
			 *     4  6  8 16 19  4
			 *     9 15  3 18 27 15
			 *    14  9  3  7 17  3
			 * Well, "9 15 3 18 27 15" didn't quite get 32 bits diffing for
			 * "differ" defined as + with a one-bit base and a two-bit delta.  I
			 * used http://burtleburtle.net/bob/hash/avalanche.html to choose
			 * the operations, constants, and arrangements of the variables.
			 *
			 * This does not achieve avalanche.  There are input bits of (a,b,c)
			 * that fail to affect some output bits of (a,b,c), especially of a.
			 * The most thoroughly mixed value is c, but it doesn't really even
			 * achieve avalanche in c.
			 *
			 * This allows some parallelism.  Read-after-writes are good at doubling
			 * the number of bits affected, so the goal of mixing pulls in the
			 * opposite direction as the goal of parallelism.  I did what I could.
			 * Rotates seem to cost as much as shifts on every machine I could lay
			 * my hands on, and rotates are much kinder to the top and bottom bits,
			 * so I used rotates.
			 */
			a -= c;
			a ^= rotateLeft(c, 4);
			c += b;

			b -= a;
			b ^= rotateLeft(a, 6);
			a += c;

			c -= b;
			c ^= rotateLeft(b, 8);
			b += a;

			a -= c;
			a ^= rotateLeft(c, 16);
			c += b;

			b -= a;
			b ^= rotateLeft(a, 19);
			a += c;

			c -= b;
			c ^= rotateLeft(b, 4);
			b += a;
		}

		// last block: affect all 32 bits of (c)
		byte[] remainingBytes = new byte[byteBuffer.remaining()];
		byteBuffer.get(remainingBytes);
		// Intentional fall-through
		switch (remainingBytes.length) {
			case 12: // NOSONAR Intentional fall-through
				c += toUnsignedInt(remainingBytes[11]) << 24;
			case 11: // NOSONAR Intentional fall-through
				c += toUnsignedInt(remainingBytes[10]) << 16;
			case 10: // NOSONAR Intentional fall-through
				c += toUnsignedInt(remainingBytes[9]) << 8;
			case 9: // NOSONAR Intentional fall-through
				c += toUnsignedInt(remainingBytes[8]);
			case 8: // NOSONAR Intentional fall-through
				b += toUnsignedInt(remainingBytes[7]) << 24;
			case 7: // NOSONAR Intentional fall-through
				b += toUnsignedInt(remainingBytes[6]) << 16;
			case 6: // NOSONAR Intentional fall-through
				b += toUnsignedInt(remainingBytes[5]) << 8;
			case 5: // NOSONAR Intentional fall-through
				b += toUnsignedInt(remainingBytes[4]);
			case 4: // NOSONAR Intentional fall-through
				a += toUnsignedInt(remainingBytes[3]) << 24;
			case 3: // NOSONAR Intentional fall-through
				a += toUnsignedInt(remainingBytes[2]) << 16;
			case 2: // NOSONAR Intentional fall-through
				a += toUnsignedInt(remainingBytes[1]) << 8;
			case 1: // NOSONAR Intentional fall-through
				a += toUnsignedInt(remainingBytes[0]);
				break;
			case 0:
				return c;
			default:
				throw new AssertionError("Invalid remaining bytes length");
		}

		return finalMix(a, b, c);
	}

}
