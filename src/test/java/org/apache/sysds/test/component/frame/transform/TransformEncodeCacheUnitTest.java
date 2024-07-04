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
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.transform.encode.*;
import org.apache.sysds.test.TestUtils;
import org.junit.After;
import org.junit.BeforeClass;
import org.junit.Test;
import java.util.LinkedList;
import java.util.Map;
import java.util.stream.IntStream;
import static org.junit.Assert.assertEquals;

public class TransformEncodeCacheUnitTest {
    protected static final Log LOG = LogFactory.getLog(TransformEncodeCacheUnitTest.class.getName());
    protected static LinkedList<EncodeCacheKey> _evicQueue = null;
    protected static Map<EncodeCacheKey, EncodeCacheEntry> _cacheMap = null;

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

            EncodeBuildCache.setCacheLimit(0.000005); // set to a lower bound for testing
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
    public void testPutEvictionLRU_noEviction() {

        int rowCount = 10;
        FrameBlock testData = TestUtils.generateRandomFrameBlock(rowCount, new Types.ValueType[]{
                        Types.ValueType.FP32, Types.ValueType.FP32, Types.ValueType.FP32}, 231);

        int columnCount = 3;
        IntStream columnIds = IntStream.range(0, columnCount);

        columnIds.forEach( id -> {
                ColumnEncoderRecode encoder = new ColumnEncoderRecode(id + 1 ); //column ids start from 1
                encoder.build(testData);
                EncodeCacheKey key = _evicQueue.getLast();
                LOG.debug(String.format("Size of entry %d: %d%n", id, _cacheMap.get(key).getSize()));
        });

        // no eviction: all build results are written into the cache
        assertEquals(columnCount, _evicQueue.size());
        assertEquals(columnCount, _cacheMap.size());
    }

    @Test
    public void testPutEvictionLRU_Eviction() {

        int rowCount = 100;
        FrameBlock testData = TestUtils.generateRandomFrameBlock(rowCount, new Types.ValueType[]{
                Types.ValueType.FP32, Types.ValueType.FP32, Types.ValueType.FP32, Types.ValueType.FP32}, 231);

        int columnCount = 4;
        IntStream columnIds = IntStream.range(0, columnCount);

        columnIds.forEach( id -> {
            ColumnEncoderRecode encoder = new ColumnEncoderRecode(id + 1 ); //column ids start from 1
            encoder.build(testData);
            EncodeCacheKey key = _evicQueue.getLast(); // the put method adds new keys to the end of the eviction queue
            LOG.debug(String.format("Size of entry %d: %d%n", id, _cacheMap.get(key).getSize()));
        });

        // eviction: evicting an entry when the cache limit is reached
        assertEquals(columnCount - 1, _evicQueue.size());
        assertEquals(columnCount - 1, _cacheMap.size());
    }

    @Test
    public void testPut_largeEntry() {

        int rowCount = 5000;
        FrameBlock testData = TestUtils.generateRandomFrameBlock(rowCount, new Types.ValueType[]{
                Types.ValueType.FP32}, 231);
        ColumnEncoderRecode encoder = new ColumnEncoderRecode(1 ); //column ids start from 1
        encoder.build(testData);

        // the build result is attempted to put into the cache but is rejected
        assertEquals(0, _evicQueue.size());
        assertEquals(0, _cacheMap.size());
    }

    @Test
    public void testGet() {

        int rowCount = 10;
        FrameBlock testData = TestUtils.generateRandomFrameBlock(rowCount, new Types.ValueType[]{
                Types.ValueType.FP32}, 231);
        ColumnEncoderRecode encoder = new ColumnEncoderRecode(1 ); //column ids start from 1
        encoder.build(testData);
        Map<Object, Long> rcdMapInEncoder = encoder.getRcdMap();
        EncodeCacheKey key = _evicQueue.getLast();

        // the build result created in the container is the same object that is retrieved
        // from the cache with the respective key
        assertEquals(rcdMapInEncoder, ((RCDMap)_cacheMap.get(key).getValue()).get_rcdMap());
    }
}
