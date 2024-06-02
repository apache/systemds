package org.apache.sysds.runtime.transform.encode;

import java.util.ArrayList;
import java.util.HashMap;

public class EncodeBuildCache {

    private HashMap<Integer, ArrayList<ColumnEncoder>> cacheMap;

    public EncodeBuildCache() {
        this.cacheMap = new HashMap<Integer, ArrayList<ColumnEncoder>>();
    }

    public HashMap<Integer, ArrayList<ColumnEncoder>> getCacheMap() {
        return cacheMap;
    }

    public void setCacheMap(HashMap<Integer, ArrayList<ColumnEncoder>> cacheMap) {
        this.cacheMap = cacheMap;
    }
}
