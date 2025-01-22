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


package org.apache.sysds.runtime.util;

public class PhiloxUniformCBPRNGenerator extends CounterBasedPRNGenerator {

    // Constants for Philox
    private static final long PHILOX_M4x64_0_hi = 0xD2E7470EE14C6C93L >>> 32;
    private static final long PHILOX_M4x64_0_lo = 0xD2E7470EE14C6C93L & 0xFFFFFFFFL;
    private static final long PHILOX_M4x64_1_hi = 0xCA5A826395121157L >>> 32;
    private static final long PHILOX_M4x64_1_lo = 0xCA5A826395121157L & 0xFFFFFFFFL;
    private static final long PHILOX_W64_0 = 0x9E3779B97F4A7C15L;
    private static final long PHILOX_W64_1 = 0xBB67AE8584CAA73BL;
    private static final double LONG_TO_01 = 0.5 / Long.MAX_VALUE;

    // Default number of rounds
    private static final int PHILOX4x64_DEFAULT_ROUNDS = 10;
    private long[] seed;

    public void setSeed(long sd) {
        this.seed = new long[2];
        this.seed[0] = sd;
        this.seed[1] = sd;
    }

    /**
     * Generate a sequence of random doubles using the Philox4x64 counter-based PRNG.
     *
     * @param ctr  The start counter to use for the PRNG
     * @param size The number of doubles to generate
     * @return An array of random doubles distributed uniformly between 0 and 1
     */
    public double[] getDoubles(long[] ctr, int size) {
        // Ensure the key is correct size
        if (this.seed.length != 2) {
            throw new IllegalArgumentException("Key must be 128 bits");
        }
        // Ensure the counter is correct size
        if (ctr.length != 4) {
            throw new IllegalArgumentException("Counter must be 256 bits");
        }

        int iterations = size / 4;
        long[] result = new long[size];
        long[] currentKey = new long[]{this.seed[0], this.seed[1]}; // Create a copy of the key

        // Reusable arrays for counters
        long[] currentCtr = ctr.clone();

        for (int i = 0; i < iterations; i++) {
            for (int j = 0; j < PHILOX4x64_DEFAULT_ROUNDS; j++) {
                // Multiply as 128-bit
                long bHigh = currentCtr[0] >>> 32;
                long bLow = currentCtr[0] & 0xFFFFFFFFL;

                long hi0 = PHILOX_M4x64_0_hi * bHigh;
                long mid1 = PHILOX_M4x64_0_hi * bLow;
                long mid2 = PHILOX_M4x64_0_lo * bHigh;
                long lo0 = PHILOX_M4x64_0_lo * bLow;

                // Combine results
                long carry = (lo0 >>> 32) + (mid1 & 0xFFFFFFFFL) + (mid2 & 0xFFFFFFFFL);
                hi0 += (mid1 >>> 32) + (mid2 >>> 32) + (carry >>> 32);
                lo0 = (lo0 & 0xFFFFFFFFL) | (carry << 32);

                // Multiply as 128-bit
                bHigh = currentCtr[2] >>> 32;
                bLow = currentCtr[2] & 0xFFFFFFFFL;

                long hi1 = PHILOX_M4x64_1_hi * bHigh;
                mid1 = PHILOX_M4x64_1_hi * bLow;
                mid2 = PHILOX_M4x64_1_lo * bHigh;
                long lo1 = PHILOX_M4x64_1_lo * bLow;

                // Combine results
                carry = (lo1 >>> 32) + (mid1 & 0xFFFFFFFFL) + (mid2 & 0xFFFFFFFFL);
                hi1 += (mid1 >>> 32) + (mid2 >>> 32) + (carry >>> 32);
                lo1 = (lo1 & 0xFFFFFFFFL) | (carry << 32);

                currentCtr[0] = hi1 ^ currentCtr[1] ^ currentKey[0];
                currentCtr[2] = hi0 ^ currentCtr[3] ^ currentKey[1];
                currentCtr[1] = lo1;
                currentCtr[3] = lo0;

                currentKey[0] += PHILOX_W64_0;
                currentKey[1] += PHILOX_W64_1;
            }

            // Unpack the results
            result[i * 4] = currentCtr[0];
            result[i * 4 + 1] = currentCtr[1];
            result[i * 4 + 2] = currentCtr[2];
            result[i * 4 + 3] = currentCtr[3];

            // Increment the counter
            if (++ctr[0] == 0 && ++ctr[1] == 0 && ++ctr[2] == 0) {
                ++ctr[3];
            }
            currentCtr[0] = ctr[0];
            currentCtr[1] = ctr[1];
            currentCtr[2] = ctr[2];
            currentCtr[3] = ctr[3];
            currentKey[0] = this.seed[0];
            currentKey[1] = this.seed[1];
        }

        // Handle remaining elements
        if (size % 4 != 0) {
            for (int j = 0; j < PHILOX4x64_DEFAULT_ROUNDS; j++) {
                // Multiply as 128-bit
                long bHigh = currentCtr[0] >>> 32;
                long bLow = currentCtr[0] & 0xFFFFFFFFL;

                long hi0 = PHILOX_M4x64_0_hi * bHigh;
                long mid1 = PHILOX_M4x64_0_hi * bLow;
                long mid2 = PHILOX_M4x64_0_lo * bHigh;
                long lo0 = PHILOX_M4x64_0_lo * bLow;

                // Combine results
                long carry = (lo0 >>> 32) + (mid1 & 0xFFFFFFFFL) + (mid2 & 0xFFFFFFFFL);
                hi0 += (mid1 >>> 32) + (mid2 >>> 32) + (carry >>> 32);
                lo0 = (lo0 & 0xFFFFFFFFL) | (carry << 32);

                // Multiply as 128-bit
                bHigh = currentCtr[2] >>> 32;
                bLow = currentCtr[2] & 0xFFFFFFFFL;

                long hi1 = PHILOX_M4x64_1_hi * bHigh;
                mid1 = PHILOX_M4x64_1_hi * bLow;
                mid2 = PHILOX_M4x64_1_lo * bHigh;
                long lo1 = PHILOX_M4x64_1_lo * bLow;

                // Combine results
                carry = (lo1 >>> 32) + (mid1 & 0xFFFFFFFFL) + (mid2 & 0xFFFFFFFFL);
                hi1 += (mid1 >>> 32) + (mid2 >>> 32) + (carry >>> 32);
                lo1 = (lo1 & 0xFFFFFFFFL) | (carry << 32);

                currentCtr[0] = hi1 ^ currentCtr[1] ^ currentKey[0];
                currentCtr[2] = hi0 ^ currentCtr[3] ^ currentKey[1];
                currentCtr[1] = lo1;
                currentCtr[3] = lo0;

                currentKey[0] += PHILOX_W64_0;
                currentKey[1] += PHILOX_W64_1;
            }

            // Store the remaining results
            switch (size % 4) {
                case 3:
                    result[iterations * 4 + 2] = currentCtr[2];
                case 2:
                    result[iterations * 4 + 1] = currentCtr[1];
                case 1:
                    result[iterations * 4] = currentCtr[0];
            }
        }
        double[] double_result = new double[result.length];
        for (int i = 0; i < result.length; i++) {
            double_result[i] = result[i] * LONG_TO_01 + .5;
        }
        return double_result;
    }
}
