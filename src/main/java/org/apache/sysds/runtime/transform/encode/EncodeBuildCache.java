package org.apache.sysds.runtime.transform.encode;

import java.util.HashMap;

public class EncodeBuildCache {

    // The integer key is the column ID, and the value is the ColumnEncoder for the given columnID
    private HashMap<Integer, ColumnEncoder> cacheMap;
    private Integer max = 20;

    public EncodeBuildCache() {
        this.cacheMap = new HashMap<Integer, ColumnEncoder>();
    }

    public HashMap<Integer, ColumnEncoder> getCacheMap() {
        return cacheMap;
    }

    public void setCacheMap(HashMap<Integer, ColumnEncoder> cacheMap) {
        this.cacheMap = cacheMap;
    }

    public void put(Integer columnID, ColumnEncoder encoder) {
        // TODO: if cachemap.size is at max, evict the least recently used column
        cacheMap.put(columnID, encoder);
    }

    public ColumnEncoder get(Integer columnID) {
        return cacheMap.get(columnID);
    }

    
}
