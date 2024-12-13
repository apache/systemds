package org.apache.sysds.test.functions.io.cog;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public abstract class COGTestBase extends AutomatedTestBase {
    protected final static String TEST_DIR = "functions/io/cog/";
    protected static final Log LOG = LogFactory.getLog(COGTestBase.class.getName());
    protected final static double eps = 1e-9;

    protected abstract String getTestClassDir();

    protected abstract String getTestName();

    protected abstract int getScriptId();

    @Override
    public void setUp() {
        addTestConfiguration(getTestName(),
                new TestConfiguration(getTestClassDir(), getTestName(), new String[] {"Rout"}));
    }
}
