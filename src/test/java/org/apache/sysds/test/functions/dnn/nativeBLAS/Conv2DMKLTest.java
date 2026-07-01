package org.apache.sysds.test.functions.dnn.nativeBLAS;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.File;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.functions.dnn.Conv2DTest;
import org.apache.sysds.utils.NativeHelper;

public class Conv2DMKLTest extends Conv2DTest {
    private final static String TEST_NAME = "Conv2DTest";
	private final static String TEST_DIR = "functions/dnn/nativeBLAS/";
	private final static String TEST_CLASS_DIR = TEST_DIR + Conv2DTest.class.getSimpleName() + "/";

    @Override
	protected File getConfigTemplateFile() {
		return new File("./src/test/config/SystemDS-config-MKL.xml");
	}

	@Override
	public void setUp() {
        try {
            TestUtils.clearAssertionInformation();
            addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));

            loadTestConfiguration(getTestConfiguration(TEST_NAME));
            DMLConfig conf = new DMLConfig(getCurConfigFile().getPath());
            ConfigurationManager.setLocalConfig(conf);
            assertEquals(true, NativeHelper.isNativeLibraryLoaded());
        } catch (Exception e) {
            e.printStackTrace();
			fail(e.getMessage());
        }
	}

    @Override
	public void tearDown() {
		TestUtils.clearDirectory(getCurLocalTempDir().getPath());
		TestUtils.removeDirectories(new String[]{getCurLocalTempDir().getPath()});
	}
}
