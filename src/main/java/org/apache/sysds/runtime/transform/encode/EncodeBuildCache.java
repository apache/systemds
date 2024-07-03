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
    private final Map<EncodeCacheKey, EncodeCacheEntry<Object>> _cache;
    private static EncodeCacheConfig.EncodeCachePolicy _cachePolicy;
    private static LinkedList<EncodeCacheKey> _evictionQueue;
    private static long _cacheLimit; //TODO: pull from yaml config
    private static long _usedCacheMemory;
    // Note: we omitted maintaining a timestamp since and ordered data structure is sufficient for LRU


    private EncodeBuildCache() {
        _cache = new ConcurrentHashMap<>();
        _cachePolicy = EncodeCacheConfig._cachepolicy;
        _evictionQueue = new LinkedList<>();
        _cacheLimit = setCacheLimit(EncodeCacheConfig.CPU_CACHE_FRAC); //5%
        _usedCacheMemory = 0;

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

    public synchronized void put(EncodeCacheKey key, EncodeCacheEntry buildResult) {
        switch (_cachePolicy) {
            case LRU:
                evictLRU(key, buildResult);
                break;
            default:
                throw new IllegalArgumentException("Unknown cache policy: " + _cachePolicy);
        }
    }

    public synchronized EncodeCacheEntry get(EncodeCacheKey key) {
        if (_cache.get(key) != null){
            LOG.debug(String.format("Getting %s from the cache\n", key));
        }

        switch (_cachePolicy) {
            case LRU:
                updateQueueLRU(key);
                break;
            default:
                throw new IllegalArgumentException("Unknown cache policy: " + _cachePolicy);
        }
        return _cache.get(key);
    }

    private synchronized void evictLRU(EncodeCacheKey key, EncodeCacheEntry buildResult){
        long entrySize = buildResult.getSize();
        long freeMemory = _cacheLimit - _usedCacheMemory;

        while (freeMemory < entrySize) {
            EncodeCacheKey evictedKey = _evictionQueue.remove();
            EncodeCacheEntry<Object> evictedEntry = _cache.get(evictedKey);
            _cache.remove(evictedKey);
            _usedCacheMemory -= evictedEntry.getSize();
            freeMemory = _cacheLimit - _usedCacheMemory;
        }

        _cache.put(key, buildResult);
        _evictionQueue.add(key);
        _usedCacheMemory += buildResult.getSize();

        LOG.debug(String.format("Putting %s in the cache\n", key));
    }

    private synchronized void updateQueueLRU(EncodeCacheKey key){
        _evictionQueue.remove(key);
        _evictionQueue.add(key);
    }

    protected static long setCacheLimit(double fraction) {
        long maxMem = InfrastructureAnalyzer.getLocalMaxMemory();
        long limit = (long)(fraction * maxMem);
        LOG.debug("Cache limit: "+ limit);
        return limit;
    }

}
