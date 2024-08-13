package org.apache.sysds.resource.enumeration;

import java.util.*;

public class GridBasedEnumerator extends AnEnumerator {
    private final int stepSizeExecutors;
    public GridBasedEnumerator(Builder builder, int stepSizeExecutors) {
        super(builder);
        this.stepSizeExecutors = stepSizeExecutors;
    }

    /**
     * Initializes the pool for driver and executor
     * instances parsed at processing with all the
     * available instances
     */
    @Override
    public void preprocessing() {
        driverSpace.initSpace(instances);
        executorSpace.initSpace(instances);
    }

    @Override
    public List<Integer> estimateRangeExecutors(long driverMemory, long executorMemory, int executorCores) {
        // consider the maximum level of parallelism and the given step size for increasing the number of executor
        int limitExecutors = MAX_LEVEL_PARALLELISM / executorCores;
        ArrayList<Integer> result = new ArrayList<>();
        for (int i = minExecutors; i <= Math.min(maxExecutors, limitExecutors); i+= stepSizeExecutors) {
            result.add(i);
        }
        return result;
    }
}
