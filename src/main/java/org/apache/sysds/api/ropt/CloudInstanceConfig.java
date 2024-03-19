package org.apache.sysds.api.ropt;

/**
 * This class describes the configurations of a single VM instance.
 */
public class CloudInstanceConfig {
    private final String instanceType;
    private final long memoryMB;
    private final int vCPUCores;
    private int numGPUs;
    private final double pricePerHour;

    public CloudInstanceConfig(String instanceType, long memoryMB, int vCPUCores, double pricePerHour) {
        this.instanceType = instanceType;
        this.memoryMB = memoryMB;
        this.vCPUCores = vCPUCores;
        // NOT relevant for now
        this.numGPUs = -1;
        this.pricePerHour = pricePerHour;
    }

    public long getMemoryMB() {
        return memoryMB;
    }

    public long getAvailableMemoryMB() {
        // NOTE: reconsider the usage of MEM_FACTOR
        return (long) (memoryMB/CloudClusterConfig.MEM_FACTOR);
    }

    public int getVCPUCores() {
        return vCPUCores;
    }

    public int getAvailableVCPUCores() {
        int availableCores = vCPUCores-CloudClusterConfig.RESERVED_CORES_OS;
        if (availableCores < 1) {
            throw new RuntimeException("Current instance '" + instanceType + "'has not sufficient number of vcores");
        }
        return availableCores;
    }

    // TODO: getting memory and vCores for a primary/managing instance
    // NOTE: This instance is responsible for more tasks

    public double getPricePerHour() {
        return pricePerHour;
    }

    public long getMaxMemoryPerCore() {
        return memoryMB/vCPUCores;
    }

    public String getInstanceType() {
        return instanceType;
    }
}
