package org.apache.sysds.test.functions.linearization;

import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

//additional imports
import org.apache.sysds.lops.compile.linearization.ILinearize;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.lops.BinaryScalar;
import org.apache.sysds.lops.Data;

public class ILinearizeTest extends AutomatedTestBase {

    private final String testDir = "functions/linearization/";

    @Override
	public void setUp() {
		setOutputBuffering(true);
		disableConfigFile = true;
		TestUtils.clearAssertionInformation();
   
		addTestConfiguration("test", new TestConfiguration(testDir, "test"));
	}

    @Test
    public void testLinearize_Pipeline() {

        try {
            //DMLConfig dmlconf = DMLConfig.readConfigurationFile("/home/niklas/systemds/src/test/scripts/functions/linearization/SystemDS-config-na-pipeline.xml");
            DMLConfig dmlconf = DMLConfig.readConfigurationFile("/home/niklas/systemds/src/test/scripts/functions/linearization/SystemDS-config-default.xml");
		    ConfigurationManager.setGlobalConfig(dmlconf);
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println("testLinearize_Pipeline");

        List<Lop> lops = new ArrayList<>();

        //lops.add(new Data(org.apache.sysds.common.Types.OpOpData.PERSISTENTREAD, null, null, "test", null, org.apache.sysds.common.Types.DataType.SCALAR, org.apache.sysds.common.Types.ValueType.INT32, null));
        Lop lop1 = Data.createLiteralLop(org.apache.sysds.common.Types.ValueType.INT32, "1");
        Lop lop2 = Data.createLiteralLop(org.apache.sysds.common.Types.ValueType.INT32, "2");
        lops.add(lop1);
        lops.add(lop2);
        lops.add(new BinaryScalar(lop1, lop2, org.apache.sysds.common.Types.OpOp2.PLUS, org.apache.sysds.common.Types.DataType.SCALAR, org.apache.sysds.common.Types.ValueType.INT32));

        List<Lop> result = ILinearize.linearize(lops);

        List<Lop> filtered = result.stream().filter(l -> l.getPipelineID() == -1).collect(Collectors.toList());
        if (filtered.size() > 0) {
            fail("Pipeline ID not set correctly");
        }

    }

}
