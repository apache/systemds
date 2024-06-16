package org.apache.sysds.runtime.transform.encode;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.Policy;
import com.github.benmanes.caffeine.cache.stats.CacheStats;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.nullness.qual.PolyNull;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentMap;
import java.util.function.Function;


public class EncodeBuildCache {

    // The integer key is the column ID and what encoding type, and the value is the build output for the given columnID

    private static Cache<CacheKey, Object> cache;
    private static EncodeBuildCache _instance = null;

    private EncodeBuildCache() {
        cache = Caffeine.newBuilder()
                .maximumSize(100) // Set your cache size here
                .build();
    }

    public static EncodeBuildCache getEncodeBuildCache() {
        if (_instance == null) {
            _instance = new EncodeBuildCache();
        }
        return _instance;
    }

    public void put(CacheKey cacheKey, Object buildResult) {

        cache.put(cacheKey, buildResult);
        System.out.printf("Putting %s in the cache;-)", cacheKey);
    }

    public Object get(CacheKey key) {

        if (cache.getIfPresent(key)!= null){
            System.out.printf("Getting %s from the cache;-)", key);
        }
        return cache.getIfPresent(key);
    }

    
}
