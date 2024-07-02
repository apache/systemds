package org.apache.sysds.runtime.transform.encode;

public class EncodeCacheConfig {

    public static boolean _cacheEnabled = true;

    //-------------CACHING LOGIC RELATED CONFIGURATIONS--------------//
    protected static final double CPU_CACHE_FRAC = 0.05; // 5% of JVM heap size

    //-------------EVICTION RELATED CONFIGURATIONS--------------//
    private static EncodeCachePolicy _cachepolicy = null;

    protected enum EncodeCacheStatus { //TODO: do we need this?
        EMPTY, 	//Placeholder with no data. Cannot be evicted.
        NOTCACHED, //Placeholder removed from the cache
        TOCACHE,   //To be cached in memory if reoccur
        CACHED,	//General cached data. Can be evicted.
        RELOADED;
        public boolean canEvict() {
            return this == CACHED || this == RELOADED;
        }
        }

    public enum EncodeCachePolicy {
        LRU
    }

    public static void useCache(boolean use){
        _cacheEnabled = use;
    }


}
