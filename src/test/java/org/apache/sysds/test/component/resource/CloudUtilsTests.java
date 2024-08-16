package org.apache.sysds.test.component.resource;

import org.apache.sysds.resource.AWSUtils;
import org.apache.sysds.resource.CloudInstance;
import org.apache.sysds.resource.CloudUtils.InstanceType;
import org.apache.sysds.resource.CloudUtils.InstanceSize;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.HashMap;

import static org.apache.sysds.test.component.resource.TestingUtils.assertEqualsCloudInstances;
import static org.apache.sysds.test.component.resource.TestingUtils.getSimpleCloudInstanceMap;
import static org.junit.Assert.*;

public class CloudUtilsTests {

    @Test
    public void getInstanceTypeAWSTest() {
        AWSUtils utils = new AWSUtils();

        InstanceType expectedValue = InstanceType.M5;
        InstanceType actualValue;

        actualValue = utils.getInstanceType("m5.xlarge");
        assertEquals(expectedValue, actualValue);

        actualValue = utils.getInstanceType("M5.XLARGE");
        assertEquals(expectedValue, actualValue);

        try {
            utils.getInstanceType("NON-M5.xlarge");
            fail("Throwing IllegalArgumentException was expected");
        } catch (IllegalArgumentException e) {
            // this block ensures correct execution of the test
        }
    }

    @Test
    public void getInstanceSizeAWSTest() {
        AWSUtils utils = new AWSUtils();

        InstanceSize expectedValue = InstanceSize._XLARGE;
        InstanceSize actualValue;

        actualValue = utils.getInstanceSize("m5.xlarge");
        assertEquals(expectedValue, actualValue);

        actualValue = utils.getInstanceSize("M5.XLARGE");
        assertEquals(expectedValue, actualValue);

        try {
            utils.getInstanceSize("m5.nonxlarge");
            fail("Throwing IllegalArgumentException was expected");
        } catch (IllegalArgumentException e) {
            // this block ensures correct execution of the test
        }
    }

    @Test
    public void validateInstanceNameAWSTest() {
        AWSUtils utils = new AWSUtils();

        // basic intel instance (old)
        assertTrue(utils.validateInstanceName("m5.2xlarge"));
        assertTrue(utils.validateInstanceName("M5.2XLARGE"));
        // basic intel instance (new)
        assertTrue(utils.validateInstanceName("m6i.xlarge"));
        // basic amd instance
        assertTrue(utils.validateInstanceName("m6a.xlarge"));
        // basic graviton instance
        assertTrue(utils.validateInstanceName("m6g.xlarge"));
        // invalid values
        assertFalse(utils.validateInstanceName("v5.xlarge"));
        assertFalse(utils.validateInstanceName("m5.notlarge"));
        assertFalse(utils.validateInstanceName("m5xlarge"));
        assertFalse(utils.validateInstanceName(".xlarge"));
        assertFalse(utils.validateInstanceName("m5."));
    }

    @Test
    public void loadCSVFileAWSTest() throws IOException {
        AWSUtils utils = new AWSUtils();

        File tmpFile = TestingUtils.generateTmpInstanceInfoTableFile();

        HashMap<String, CloudInstance> actual = utils.loadInstanceInfoTable(tmpFile.getPath());
        HashMap<String, CloudInstance> expected = getSimpleCloudInstanceMap();

        for (String instanceName: expected.keySet()) {
            assertEqualsCloudInstances(expected.get(instanceName), actual.get(instanceName));
        }

        Files.deleteIfExists(tmpFile.toPath());
    }
}
