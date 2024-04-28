package org.apache.sysds.api.ropt.old_impl;


public class CloudClusterConfig {
    // TODO: This of a real strategy for choosing the constants down
    public static final double MEM_FACTOR = 1.5;
    public static final int RESERVED_CORES_OS = 1;
    // Instance to host the managing daemons - cluster manager (YARN resource manager) and Name Node (HDFS manager)
    private final CloudInstanceConfig managingInstance;
    // Instance to executes the Spark Driver and by that the Control Program
    private CloudInstanceConfig cpInstance;
    // Instances to execute the Spark Jobs
    private CloudInstanceConfig spGroupInstance;
    private int spGroupSize;

    public CloudClusterConfig(CloudInstanceConfig managingInstance, CloudInstanceConfig cpInstance)
    {
        this.managingInstance = managingInstance;
        this.cpInstance = cpInstance;
        this.spGroupSize = 0;
        this.spGroupInstance = null;
    }

    public CloudInstanceConfig getManagingInstance() {
        return managingInstance;
    }

    public CloudInstanceConfig getCpInstance() {
        return cpInstance;
    }

    public CloudInstanceConfig getSpGroupInstance() {
        return spGroupInstance;
    }

    public int getSpGroupSize() { return spGroupSize; }

    public void setCpInstance(CloudInstanceConfig cpInstance) {
        this.cpInstance = cpInstance;
    }

    public void setSpGroupInstance(CloudInstanceConfig spGroupInstance) {
        setSpGroupInstance(1, spGroupInstance);
    }

    public void setSpGroupInstance(int spGroupSize, CloudInstanceConfig spGroupInstance) {
        this.spGroupSize = spGroupSize;
        this.spGroupInstance = spGroupInstance;
    }

    public void setSpGroupSize(int spGroupSize) {
        this.spGroupSize = spGroupSize;
    }

    public double getPricePerHour() {
        return (managingInstance.getPricePerHour() +
                cpInstance.getPricePerHour() +
                spGroupSize*spGroupInstance.getPricePerHour());
    }

    public double getFinalPrice(double time) {
        // transfer seconds to hour and ceil the result (per hour price)
        double hours = Math.ceil(time/3600.0);
        // NOTE(1): think of adding extra puffer to encounter imprecise cost estimation
        // NOTE(2): think of additional costs (networking, etc.)
        return hours*getPricePerHour();
    }

}
