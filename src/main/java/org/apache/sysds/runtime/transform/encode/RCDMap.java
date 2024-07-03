package org.apache.sysds.runtime.transform.encode;

import java.util.Map;

public class RCDMap {

    private final Map<Object, Long> _rcdMap;
    private final long size;

    public RCDMap(Map<Object, Long> _rcdMap, long size) {
        this._rcdMap = _rcdMap;
        this.size = size;
    }

    public Map<Object, Long> get_rcdMap() {
        return _rcdMap;
    }

    public long getSize() {
        return size;
    }

    @Override
    public String toString() {
        return "RCDMap{" +
                "_rcdMap=" + _rcdMap +
                ", size=" + size +
                '}';
    }
}
