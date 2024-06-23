package org.apache.sysds.runtime.transform.encode;

public class BinMinsMaxs {

    private final double[] _binMins;
    private final double[] _binMaxs;

    public BinMinsMaxs(double[] binMins, double[] binMaxs) {
        this._binMins = binMins;
        this._binMaxs = binMaxs;
    }

    public double[] get_binMaxs() {
        return _binMaxs;
    }

    public double[] get_binMins() {
        return _binMins;
    }
    
}
