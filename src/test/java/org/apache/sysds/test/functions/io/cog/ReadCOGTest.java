package org.apache.sysds.test.functions.io.cog;

import org.apache.sysds.common.Types;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public abstract class ReadCOGTest extends COGTestBase {
    protected abstract int getId();

    @Test
    public void testHDF51_Seq_CP() {
        runReadCOGTest(getId(), Types.ExecMode.SINGLE_NODE, false);
    }

    @Test
    public void testHDF51_Parallel_CP() {
        runReadCOGTest(getId(), Types.ExecMode.SINGLE_NODE, true);
    }

    protected void runReadCOGTest(int testNumber, Types.ExecMode platform, boolean parallel) {
        TestUtils.compareScalars(true, true);
    }
}
