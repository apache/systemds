package org.apache.sysds.resource;

import org.apache.sysds.resource.enumeration.EnumerationUtils;

public class AWSUtils extends CloudUtils {
    public static final String EC2_REGEX = "^([a-z]+)([0-9])(a|g|i?)([bdnez]*)\\.([a-z0-9]*)$";
    @Override
    public boolean validateInstanceName(String instanceName) {
        return instanceName.matches(EC2_REGEX);
    }

    @Override
    public InstanceType getInstanceType(String instanceName) {
        String typeAsString = instanceName.split("\\.")[0];
        // throws exception if string value is not valid
        return InstanceType.customValueOf(typeAsString);
    }

    @Override
    public InstanceSize getInstanceSize(String instanceName) {
        String sizeAsString = instanceName.split("\\.")[1];
        // throws exception if string value is not valid
        return InstanceSize.customValueOf(sizeAsString);
    }

    @Override
    public double calculateClusterPrice(EnumerationUtils.ConfigurationPoint config, double time) {
        double pricePerSeconds = getClusterCostPerHour(config);
        return (DEFAULT_CLUSTER_LAUNCH_TIME + time) * pricePerSeconds;
    }

    private double getClusterCostPerHour(EnumerationUtils.ConfigurationPoint config) {
        if (config.numberExecutors == 0) {
            return config.driverInstance.getPrice();
        }
        return config.driverInstance.getPrice() +
                config.executorInstance.getPrice()*config.numberExecutors;
    }
}
