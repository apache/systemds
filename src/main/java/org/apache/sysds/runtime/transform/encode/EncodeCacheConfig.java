package org.apache.sysds.runtime.transform.encode;

public class EncodeCacheConfig {

    public static boolean _cacheEnabled = true;

    protected static final double CPU_CACHE_FRAC = 0.05; // 5% of JVM heap size

    public enum EncodeCachePolicy {
        LRU
    }

    public static EncodeCachePolicy _cachepolicy = EncodeCachePolicy.LRU;

    public static void useCache(boolean use){
        _cacheEnabled = use;
    }

}
