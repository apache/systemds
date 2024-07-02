package org.apache.sysds.runtime.transform.encode;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;


import java.util.LinkedList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;


public class EncodeBuildCache {

    protected static final Log LOG = LogFactory.getLog(EncodeBuildCache.class.getName());
    private static volatile EncodeBuildCache _instance;
    private final Map<EncodeCacheKey, EncodeCacheEntry> _cache;
    private static LinkedList<EncodeCacheKey> _evictionQueue;
    private static long _cacheSize;
    private static long _cacheLimit; //TODO: pull from yaml config

    private static long _maxCacheMemory;

    private static long _usedCacheMemory;

    private static long _startTimestamp;

    private EncodeBuildCache() {
        _cache = new ConcurrentHashMap<>();
        _evictionQueue = new LinkedList<>();
        _cacheLimit = setCacheLimit(EncodeCacheConfig.CPU_CACHE_FRAC); //5%
        _maxCacheMemory = _cacheLimit*Runtime.getRuntime().totalMemory();
        _usedCacheMemory = 0;
        _startTimestamp = System.currentTimeMillis(); //TODO: do we need it?
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

    public void put(EncodeCacheKey key, EncodeCacheEntry buildResult) {


        long entrySize = buildResult.getSize();
        long freeMemory = _maxCacheMemory - _usedCacheMemory;

        while (freeMemory < entrySize) {
            EncodeCacheKey evictedKey = _evictionQueue.remove();
            EncodeCacheEntry evictedEntry = _cache.get(evictedKey);
            _cache.remove(evictedKey);
            _usedCacheMemory -= evictedEntry.getSize();
            freeMemory = _maxCacheMemory - _usedCacheMemory;
        }

        _cache.put(key, buildResult);
        _evictionQueue.add(key);
        _usedCacheMemory += buildResult.getSize();

        LOG.debug(String.format("Putting %s in the cache\n", key));
    }

    public EncodeCacheEntry get(EncodeCacheKey key) {

        //TODO: update timestamp in the newly used cache entry
        //TODO: move newly used key to the top of the eviction queue
        if (_cache.get(key) != null){
            LOG.debug(String.format("Getting %s from the cache\n", key));
        }
        return _cache.get(key);
    }

    protected static long setCacheLimit(double fraction) {
        long maxMem = InfrastructureAnalyzer.getLocalMaxMemory();
        long limit = (long)(fraction * maxMem);
        LOG.debug("Cache limit: "+ limit);
        return limit;
    }



}
