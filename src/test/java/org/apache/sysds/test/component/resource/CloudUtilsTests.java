package org.apache.sysds.test.component.resource;

import org.apache.sysds.resource.AWSUtils;
import org.apache.sysds.resource.CloudInstance;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import static org.apache.sysds.test.component.resource.TestingUtils.assertEqualsCloudInstances;
import static org.apache.sysds.test.component.resource.TestingUtils.getSimpleCloudInstanceMap;

public class CloudUtilsTests {
    @Test
    public void loadCSVFileTest() throws IOException {
        File tmpFile = File.createTempFile("systemds_tmp", ".csv");

        List<String> csvLines = Arrays.asList(
                "API_Name,Memory,vCPUs,gFlops,ramSpeed,diskSpeed,networkSpeed,Price",
                "m5.xlarge,16.0,4,0.5,0,143.75,160,1.5",
                "m5.2xlarge,32.0,8,1.0,0,0,0,1.9",
                "c5.xlarge,8.0,4,0.5,0,0,0,1.7",
                "c5.2xlarge,16.0,8,1.0,0,0,0,2.1"
        );
        Files.write(tmpFile.toPath(), csvLines);

        AWSUtils utils = new AWSUtils();
        HashMap<String, CloudInstance> actual = utils.loadInstanceInfoTable(tmpFile.getPath());
        HashMap<String, CloudInstance> expected = getSimpleCloudInstanceMap();

        for (String instanceName: expected.keySet()) {
            assertEqualsCloudInstances(expected.get(instanceName), actual.get(instanceName));
        }

        Files.deleteIfExists(tmpFile.toPath());
    }
}
