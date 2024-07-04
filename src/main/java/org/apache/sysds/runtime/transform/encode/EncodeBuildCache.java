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

package org.apache.sysds.runtime.transform.encode;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;

import java.util.LinkedList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;


public class EncodeBuildCache {

    protected static final Log LOG = LogFactory.getLog(EncodeBuildCache.class.getName());
    private static volatile EncodeBuildCache _instance;
    private static Map<EncodeCacheKey, EncodeCacheEntry> _cache = null;

    private final EncodeCacheConfig _config;
    private static EncodeCacheConfig.EncodeCachePolicy _cachePolicy;
    private static LinkedList<EncodeCacheKey> _evictionQueue;
    private static long _cacheLimit;
    private static long _usedCacheMemory;
    // Note: we omitted maintaining a timestamp since and ordered data structure is sufficient for LRU

    private EncodeBuildCache() {
        _cache = new ConcurrentHashMap<>();
        _cachePolicy = EncodeCacheConfig._cachepolicy;
        _evictionQueue = new LinkedList<>();
        _usedCacheMemory = 0;
        _config = EncodeCacheConfig.create(); // reads values from ConfigurationManager or uses default ones
        setCacheLimit(_config.getCacheMemoryFraction()); //5%
    }

    // we chose the singleton pattern instead of making the cache a static class because it is lazy loaded
    public static EncodeBuildCache getEncodeBuildCache() {
        if (_instance == null) {
            synchronized (EncodeBuildCache.class) {
                if (_instance == null) {
                    _instance = new EncodeBuildCache();
                }
            }
        }
        return _instance;
    }

    public static void clear() {
        try {
            _evictionQueue.clear();
            _cache.clear();
            _usedCacheMemory = 0;
        } catch(Exception e) {
            LOG.error("Encode cache instance could not be cleared partially or completely. Cause: " + e.getMessage());
        }


    }

    public void put(EncodeCacheKey key, EncodeCacheEntry buildResult) {
        long entrySize = buildResult.getSize();
        long freeMemory = _cacheLimit - _usedCacheMemory;
        LOG.debug(String.format("Free memory in cache: %d", freeMemory));

        if (entrySize > _cacheLimit) {
            LOG.info(String.format("Size of build result exceeds cache limit. Size: %d ", buildResult.getSize()));
            return;
        }

        if (freeMemory < entrySize) {
            evict(key, buildResult);
        }
        _cache.put(key, buildResult);
        _evictionQueue.add(key);
        _usedCacheMemory += buildResult.getSize();
        LOG.info(String.format("Putting entry with size %d in cache: %s\n", buildResult.getSize(), buildResult));
    }

    public EncodeCacheEntry get(EncodeCacheKey key) {
        try {
            EncodeCacheEntry entry = _cache.get(key);

            if (entry != null) {
                LOG.debug(String.format("Getting %s from the cache\n", key));

                switch (_cachePolicy) {
                    case LRU:
                        updateQueueLRU(key);
                        break;
                }
                return entry;
            }

        } catch (Exception e) {
            LOG.error("Unexpected error occurred: " + e.getMessage(), e);
        }
        return null;
    }

    private void evict(EncodeCacheKey key, EncodeCacheEntry buildResult) {
        switch (_cachePolicy) {
            case LRU:
                evictLRU(key, buildResult);
                break;
        }
    }

    private void evictLRU(EncodeCacheKey key, EncodeCacheEntry buildResult) {
        long entrySize = buildResult.getSize();

        long freeMemory = _cacheLimit - _usedCacheMemory;

        while (freeMemory < entrySize) {
            try {
                System.out.println("Evicting using LRU policy..");
                EncodeCacheKey evictedKey = _evictionQueue.remove();
                EncodeCacheEntry<Object> evictedEntry = _cache.get(evictedKey);
                _cache.remove(evictedKey);
                _usedCacheMemory -= evictedEntry.getSize();
                LOG.debug(String.format("Used memory: %d", _usedCacheMemory));
                freeMemory = _cacheLimit - _usedCacheMemory;

            } catch (Exception e) {
                throw new DMLRuntimeException(e.getMessage());
            }
        }
    }

    private void updateQueueLRU(EncodeCacheKey key) {
        _evictionQueue.remove(key);
        _evictionQueue.add(key);
    }

    public static void setCacheLimit(double fraction) { // this is void now for manipulating cache limit in the test
        long maxMem = InfrastructureAnalyzer.getLocalMaxMemory();
        long limit = (long) (fraction * maxMem);
        LOG.debug("Set cache limit to: " + limit);
        _cacheLimit = limit;
    }

    // ------ methods used for testing --------

    public static long get_cacheLimit() {
        return _cacheLimit;
    }

    public static long get_usedCacheMemory() {
        return _usedCacheMemory;
    }

    public static Map<EncodeCacheKey, EncodeCacheEntry> get_cache() {
        return _cache;
    }

    public static LinkedList<EncodeCacheKey> get_evictionQueue() {
        return _evictionQueue;
    }
}
