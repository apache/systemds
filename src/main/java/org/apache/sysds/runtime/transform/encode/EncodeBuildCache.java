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
    private static LinkedList<EncodeCacheEntry> _evictionQueue;
    private static long _cacheSize;
    private static long _cacheLimit; //TODO: pull from yaml config
    private static long _startTimestamp;

    private EncodeBuildCache() {
        _cache = new ConcurrentHashMap<>();
        _evictionQueue = new LinkedList<>();
        _cacheSize = 0;
        _cacheLimit = setCacheLimit(EncodeCacheConfig.CPU_CACHE_FRAC); //5%
        _startTimestamp = System.currentTimeMillis(); //TODO: do we need it?
    }
    // we chose the singleton pattern instead of making the cache a static class because it is lazy loaded
    public static EncodeBuildCache getInstance() {
        if (_instance == null) {
            synchronized (EncodeBuildCache.class) {
                if (_instance == null) {
                    _instance = new EncodeBuildCache();
                }
            }
        }
        return _instance;
    }

    public static EncodeBuildCache getEncodeBuildCache() {
        if (_instance == null) {
            _instance = new EncodeBuildCache();
        }
        return _instance;
    }

    public void put(EncodeCacheKey key, EncodeCacheEntry buildResult) {

        //TODO: check available space, evict if neccessary, delete as many objects as needed
        // the cache entry object has a getSize method that needs to be implemented
        _cache.put(key, buildResult);
        //TODO: update eviction list, too
        //TODO: update cache size
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
