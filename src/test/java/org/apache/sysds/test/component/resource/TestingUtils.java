package org.apache.sysds.test.component.resource;

import org.apache.sysds.resource.CloudInstance;
import org.junit.Assert;

import java.util.HashMap;

public class TestingUtils {
    public static void assertEqualsCloudInstances(CloudInstance expected, CloudInstance actual) {
        Assert.assertEquals(expected.getInstanceName(), actual.getInstanceName());
        Assert.assertEquals(expected.getMemory(), actual.getMemory());
        Assert.assertEquals(expected.getVCPUs(), actual.getVCPUs());
        Assert.assertEquals(expected.getFLOPS(), actual.getFLOPS());
        Assert.assertEquals(expected.getMemorySpeed(), actual.getMemorySpeed(), 0.0);
        Assert.assertEquals(expected.getDiskSpeed(), actual.getDiskSpeed(), 0.0);
        Assert.assertEquals(expected.getNetworkSpeed(), actual.getNetworkSpeed(), 0.0);
        Assert.assertEquals(expected.getPrice(), actual.getPrice(), 0.0);

    }

    public static HashMap<String, CloudInstance> getSimpleCloudInstanceMap() {
        HashMap<String, CloudInstance> instanceMap =  new HashMap<>();
        // fill the map wsearchStrategyh enough cloud instances to allow testing all search space dimension searchStrategyerations
        instanceMap.put("m5.xlarge", new CloudInstance("m5.xlarge", 16L*1024*1024*1024, 4, 0.5, 0.0, 143.75, 160, 1.5));
        instanceMap.put("m5.2xlarge", new CloudInstance("m5.2xlarge", 32L*1024*1024*1024, 8, 1.0, 0.0, 0.0, 0.0, 1.9));
        instanceMap.put("c5.xlarge", new CloudInstance("c5.xlarge", 8L*1024*1024*1024, 4, 0.5, 0.0, 0.0, 0.0, 1.7));
        instanceMap.put("c5.2xlarge", new CloudInstance("c5.2xlarge", 16L*1024*1024*1024, 8, 1.0, 0.0, 0.0, 0.0, 2.1));

        return instanceMap;
    }
}
