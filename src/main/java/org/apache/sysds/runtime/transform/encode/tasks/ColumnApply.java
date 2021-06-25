package org.apache.sysds.runtime.transform.encode.tasks;

import java.util.concurrent.Callable;

public interface ColumnApply extends Callable<Object> {
    void setOutputCol(int outputCol);
    int getOutputCol();
}
