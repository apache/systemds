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

public class PhiloxNormalCBPRNGenerator extends CounterBasedPRNGenerator {
    private long[] seed;
    private PhiloxUniformCBPRNGenerator uniformGen;

    public void setSeed(long sd) {
        this.seed = new long[2];
        this.seed[0] = sd;
        this.seed[1] = sd;
        uniformGen = new PhiloxUniformCBPRNGenerator();
        uniformGen.setSeed(this.seed[0]);
    }

    /**
     * Generate a sequence of random doubles using the Philox4x64 counter-based PRNG.
     *
     * @param ctr  The start counter to use for the PRNG
     * @param size The number of doubles to generate
     * @return An array of random doubles distributed normally with mean 0 and variance 1
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

        double[] uniform = uniformGen.getDoubles(ctr, size + size % 2);
        double[] normal = new double[size];
        for (int i = 0; i < size; i+=2) {
            double v1 = Math.sqrt(-2*Math.log(uniform[i]));
            double v2 = 2*Math.PI*uniform[i + 1];
            normal[i] = v1 * Math.cos(v2);
            if (i + 1 < size) {
                normal[i + 1] = v1 * Math.sin(v2);
            }
        }

        return normal;
    }
}
