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

package org.apache.sysds.test.component.frame.transform;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.transform.encode.*;
import org.junit.After;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.*;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;

public class TransformEncodeCacheBuildTest {
    protected static final Log LOG = LogFactory.getLog(TransformEncodeCacheBuildTest.class.getName());
    protected static LinkedList<EncodeCacheKey> _evicQueue = null;
    protected static Map<EncodeCacheKey, EncodeCacheEntry> _cacheMap = null;
    protected static int _numColumns = 10;
    protected static int _numRows = 1000;
    protected static FrameBlock _testData = EncodeCacheTestUtil.generateTestData(_numColumns, _numRows);

    @BeforeClass
    public static void setUp() {
        try {
            long st = System.nanoTime();
            EncodeBuildCache.getEncodeBuildCache();
            long et = System.nanoTime();
            double setUpTime = (et - st)/1_000_000.0;

            _evicQueue = EncodeBuildCache.get_evictionQueue();
            LOG.debug(String.format("Successfully set up cache in %f milliseconds. " +
                    "Size of eviction queue: %d", setUpTime, _evicQueue.size()));

            _cacheMap = EncodeBuildCache.get_cache();

            EncodeBuildCache.setCacheLimit(0.05); // set to a lower bound for testing
            LOG.debug(String.format("Cache limit: %d", EncodeBuildCache.get_cacheLimit()));

        } catch(DMLRuntimeException e){
            LOG.error("Creation of cache failed:" + e.getMessage());
        }
    }

    @After // runs after each test
    public void cleanUp(){
        EncodeBuildCache.clear(); // clears map and queue but does not reset cache limit
    }

    @Test
    public void compareRecodeBuildWithAndWithoutCache() {
        EncodeCacheConfig.useCache(false);
        test(false, "recode");

        EncodeCacheConfig.useCache(true);
        test(true, "recode");
    }

    @Test
    public void compareBinBuildWithAndWithoutCache() {
        EncodeCacheConfig.useCache(false);
        test(false, "bin");

        EncodeCacheConfig.useCache(true);
        test(true, "bin");
    }

    public void test(Boolean cacheUsed, String encoderType ){

        System.out.println("Cache used: " + cacheUsed.toString());
        System.out.println(encoderType);

        IntStream columnIds = IntStream.range(0, _numColumns);

        List<Long> firstBuildTimes = new ArrayList<>();
        List<Long> secondBuildTimes = new ArrayList<>();

        columnIds.forEach( id -> {
            ColumnEncoder encoder = null;
            switch (encoderType){
                case "recode":
                    encoder = new ColumnEncoderRecode(id + 1); //column ids start from 1
                    break;
                case "bin":
                    encoder = new ColumnEncoderBin(id + 1, 5, ColumnEncoderBin.BinMethod.EQUI_WIDTH );
                    break;

            }
            //System.out.println(encoder.getClass());
            firstBuildTimes.add(EncodeCacheTestUtil.measureBuildTime(encoder, _testData));
            secondBuildTimes.add(EncodeCacheTestUtil.measureBuildTime(encoder, _testData));
        });

        double avgFirstBuild = EncodeCacheTestUtil.analyzePerformance(2, firstBuildTimes, cacheUsed);
        double avgSecondBuild = EncodeCacheTestUtil.analyzePerformance(2, secondBuildTimes, cacheUsed);
        System.out.printf("Average build time in the first run: %f%n", avgFirstBuild);
        System.out.printf("Average build time in the second run: %f%n", avgSecondBuild);
    }
}
