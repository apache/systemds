package org.apache.sysds.resource;

/**
 * This class describes the configurations of a single VM instance.
 * The idea is to use this class to represent instances of different
 * cloud hypervisors - currently supporting only EC2 instances by AWS.
 */
public class CloudInstance {
    private final String instanceName;
    private final long memory;
    private final int vCPUCores;
    private final double pricePerHour;
    private final double gFlops;
    private final double memorySpeed;
    private final double diskSpeed;
    private final double networkSpeed;
    public CloudInstance(String instanceName, long memory, int vCPUCores, double gFlops, double memorySpeed, double diskSpeed, double networkSpeed, double pricePerHour) {
        this.instanceName = instanceName;
        this.memory = memory;
        this.vCPUCores = vCPUCores;
        this.gFlops = gFlops;
        this.memorySpeed = memorySpeed;
        this.diskSpeed = diskSpeed;
        this.networkSpeed = networkSpeed;
        this.pricePerHour = pricePerHour;
    }

    public String getInstanceName() {
        return instanceName;
    }

    /**
     * @return memory of the instance in B
     */
    public long getMemory() {
        return memory;
    }

    /**
     * @return number of virtual CPU cores of the instance
     */
    public int getVCPUs() {
        return vCPUCores;
    }

    /**
     * @return price per hour of the instance
     */
    public double getPrice() {
        return pricePerHour;
    }

    /**
     * @return number of FLOPS of the instance
     */
    public long getFLOPS() {
        return (long) (gFlops*1024)*1024*1024;
    }

    /**
     * @return memory speed/bandwidth of the instance in MB/s
     */
    public double getMemorySpeed() {
        return memorySpeed;
    }

    /**
     * @return isk speed/bandwidth of the instance in MB/s
     */
    public double getDiskSpeed() {
        return diskSpeed;
    }

    /**
     * @return network speed/bandwidth of the instance in MB/s
     */
    public double getNetworkSpeed() {
        return networkSpeed;
    }
}
