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

package org.apache.sysds.utils.stats;

import java.util.concurrent.atomic.LongAdder;

public class FederatedCompressionStatistics {

    private static final LongAdder totalEncodingMillis = new LongAdder();
    private static final LongAdder totalDecodingMillis = new LongAdder();
    private static final LongAdder totalEncodingBeforeSize= new LongAdder();
    private static final LongAdder totalEncodingAfterSize = new LongAdder();
    private static final LongAdder totalDecodingBeforeSize= new LongAdder();
    private static final LongAdder totalDecodingAfterSize = new LongAdder();

    public static void encodingStep(long encodingMillis, long beforeSize, long afterSize) {
        totalEncodingMillis.add(encodingMillis);
        totalEncodingBeforeSize.add(beforeSize);
        totalEncodingAfterSize.add(afterSize);
    }

    public static void decodingStep(long decodingMillis, long beforeSize, long afterSize) {
        totalDecodingMillis.add(decodingMillis);
        totalDecodingBeforeSize.add(beforeSize);
        totalDecodingAfterSize.add(afterSize);
    }

    public static void reset() {
        totalEncodingMillis.reset();
        totalDecodingMillis.reset();
        totalEncodingBeforeSize.reset();
        totalEncodingAfterSize.reset();
        totalDecodingBeforeSize.reset();
        totalDecodingAfterSize.reset();
    }

    public static String statistics() {
        StringBuilder sb = new StringBuilder();
        sb.append("Federated Compression Statistics (Worker):\n");
        sb.append("Encoding:\n");
        sb.append(" Total encoding millis: " + totalEncodingMillis.longValue() + "\n");
        sb.append(" Total pre-encoding size: " + totalEncodingBeforeSize.longValue() + "\n");
        sb.append(" Total post-encoding size: " + totalEncodingAfterSize.longValue() + "\n");
        sb.append(" Compression ratio: " + (double)(totalEncodingAfterSize.longValue())/((double)totalEncodingBeforeSize.longValue()) + "\n");
        sb.append("Decoding:\n");
        sb.append(" Total decoding millis: " + totalDecodingMillis.longValue() + "\n");
        sb.append(" Total pre-decoding size: " + totalDecodingBeforeSize.longValue() + "\n");
        sb.append(" Total post-decoding size: " + totalDecodingAfterSize.longValue() + "\n");
        sb.append(" Compression ratio: " + (double)(totalDecodingBeforeSize.longValue())/((double)totalDecodingAfterSize.longValue()) + "\n");
        return sb.toString();
    }

}
