package org.apache.sysds.resource;

import org.apache.sysds.resource.enumeration.EnumerationUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

public abstract class CloudUtils {
    public enum CloudProvider {
        AWS // potentially AZURE, GOOGLE
    }
    public enum InstanceType {
        // AWS EC2 instance
        C5, C5A, C6I, C6A, C6G, C7I, C7A, C7G, // compute optimized - vCores:mem~=1:2
        R5, R5A, R6I, R6A, R6G, R7I, R7A, R7G // memory optimized - vCores:mem~=1:8
        // Potentially VM instance types for different Cloud providers
    }

    public enum InstanceSize {
        _XLARGE, _2XLARGE, _4XLARGE, _8XLARGE, _12XLARGE, _16XLARGE, _24XLARGE, _32XLARGE, _48XLARGE
        // Potentially VM instance sizes for different Cloud providers
    }

    public static final double MINIMAL_EXECUTION_TIME = 60;

    public static final double DEFAULT_CLUSTER_LAUNCH_TIME = 120; // seconds; NOTE: set always to at least 60 seconds

    public abstract boolean validateInstanceName(String instanceName);
    public abstract InstanceType getInstanceType(String instanceName);
    public abstract InstanceSize getInstanceSize(String instanceName);

    /**
     * This method calculates the cluster price based on the
     * estimated execution time and the cluster configuration.
     * @param config the cluster configuration for the calculation
     * @param time estimated execution time in seconds
     * @return price for the given time
     */
    public abstract double calculateClusterPrice(EnumerationUtils.ConfigurationPoint config, double time);

    /**
     * Performs read of csv file filled with VM instance characteristics.
     * Each record in the csv should carry the following information (including header):
     * <li>API_Name - naming for VM instance used by the provider</li>
     * <li>Memory - floating number for the instance memory in GBs</li>
     * <li>vCPUs - number of physical threads</li>
     * <li>gFlops - FLOPS capability of the CPU in GFLOPS (Giga)</li>
     * <li>ramSpeed - memory bandwidth in MB/s</li>
     * <li>diskSpeed - memory bandwidth in MB/s</li>
     * <li>networkSpeed - memory bandwidth in MB/s</li>
     * <li>Price - price for instance per hour</li>
     * @param instanceTablePath csv file
     * @return map with filtered instances
     * @throws IOException in case problem at reading the csv file
     */
    public HashMap<String, CloudInstance> loadInstanceInfoTable(String instanceTablePath) throws IOException {
        HashMap<String, CloudInstance> result = new HashMap<>();
        int lineCount = 1;
        // try to open the file
        BufferedReader br = new BufferedReader(new FileReader(instanceTablePath));
        String parsedLine;
        // validate the file header
        parsedLine = br.readLine();
        if (!parsedLine.equals("API_Name,Memory,vCPUs,gFlops,ramSpeed,diskSpeed,networkSpeed,Price"))
            throw new IOException("Invalid CSV header inside: " + instanceTablePath);


        while ((parsedLine = br.readLine()) != null) {
            String[] values = parsedLine.split(",");
            if (values.length != 8 || !validateInstanceName(values[0]))
                throw new IOException(String.format("Invalid CSV line(%d) inside: %s", lineCount, instanceTablePath));

            String API_Name = values[0];
            long Memory = (long) (Double.parseDouble(values[1])*1024)*1024*1024;
            int vCPUs = Integer.parseInt(values[2]);
            double gFlops = Double.parseDouble(values[3]);
            double ramSpeed = Double.parseDouble(values[4]);
            double diskSpeed = Double.parseDouble(values[5]);
            double networkSpeed = Double.parseDouble(values[6]);
            double Price = Double.parseDouble(values[7]);

            CloudInstance parsedInstance = new CloudInstance(
                    API_Name,
                    Memory,
                    vCPUs,
                    gFlops,
                    ramSpeed,
                    diskSpeed,
                    networkSpeed,
                    Price
            );
            result.put(API_Name, parsedInstance);
            lineCount++;
        }

        return result;
    }
}
