package org.apache.sysds.api.ropt;


/**
 * This class describes the configurations of a single VM instance.
 */
public class CloudInstance {
    private final String instanceName;
    private final long memoryMB;
    private final int vCPUCores;
    private int numGPUs; // NOTE: to be used in the future
    private final double pricePerHour;
    private final double gFlops;

    public CloudInstance(String instanceName, long memoryMB, int vCPUCores, double pricePerHour, double gFlops) {
        if (!instanceName.matches(AWSUtils.EC2_REGEX))
            throw new RuntimeException(instanceName+" is not a valid instance name");
        this.instanceName = instanceName;
        this.memoryMB = memoryMB;
        this.vCPUCores = vCPUCores;
        // NOT relevant for now
        this.numGPUs = 0;
        this.pricePerHour = pricePerHour;
        this.gFlops = gFlops;
    }

    public String getInstanceName() {
        return instanceName;
    }

    public String getInstanceType() {
        return instanceName.split("\\.")[0];
    }

    public String getInstanceSize() {
        return instanceName.split("\\.")[1];
    }
    public long getMemoryMB() {
        return memoryMB;
    }

    public long getAvailableMemoryMB() {
        // NOTE: reconsider the usage of MEM_FACTOR
        return (long) (memoryMB/ CloudOptimizerUtils.MEM_FACTOR);
    }

    public int getVCPUCores() {
        return vCPUCores;
    }


    public double getPricePerHour() {
        return pricePerHour;
    }

    public long getFLOPS() {
        return (long) (gFlops*1024)*1024*1024;
    }

    public long getMemoryPerCore() {
        return memoryMB/vCPUCores;
    }

}
