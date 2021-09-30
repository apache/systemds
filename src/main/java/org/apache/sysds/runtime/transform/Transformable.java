package org.apache.sysds.runtime.transform;

public interface Transformable {


    int getNumRows();
    int getNumColumns();

    double getDoubleValue(int r, int c);
    String getStringValue(int r, int c);
}
