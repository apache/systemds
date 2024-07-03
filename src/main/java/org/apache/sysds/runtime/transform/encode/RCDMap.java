package org.apache.sysds.runtime.transform.encode;

import java.util.Map;

public class RCDMap {

    private final Map _rcdMap;
    private final long size;

    public RCDMap(Map _rcdMap, long size) {
        this._rcdMap = _rcdMap;
        this.size = size;
    }

    public Map get_rcdMap() {
        return _rcdMap;
    }

    public long getSize() {
        return size;
    }
}
