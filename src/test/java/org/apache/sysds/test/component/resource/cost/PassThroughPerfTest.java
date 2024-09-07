package org.apache.sysds.test.component.resource.cost;

import org.apache.sysds.resource.CloudInstance;
import org.apache.sysds.resource.ResourceCompiler;
import org.apache.sysds.resource.cost.CostEstimationException;
import org.apache.sysds.resource.cost.CostEstimator;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.utils.Explain;

import org.junit.Test;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.apache.sysds.resource.CloudUtils.GBtoBytes;
import static org.apache.sysds.test.component.resource.TestingUtils.getSimpleCloudInstanceMap;
import static org.junit.Assert.assertTrue;

/**
 * These tests check the correct worflow
 * execution of algorithms from 'scripts/perftests'
 * but do not check the result values
 */
public class PassThroughPerfTest {
    private static final HashMap<String, CloudInstance> instanceMap = getSimpleCloudInstanceMap();

    @Test
    public void alsCGCostEstimationTestSingleNode() throws IOException {
        // test successful parsing
        Map<String, String> nvargs = new HashMap<>();
        nvargs.put("$X", "tests/X.csv");
        nvargs.put("$modelU", "tests/U.txt");
        nvargs.put("$modelV", "tests/V.txt");
        ResourceCompiler.setDriverConfigurations(GBtoBytes(16), 4);
        ResourceCompiler.setSingleNodeExecution();
        Program program = ResourceCompiler.compile("scripts/perftest/scripts/alsCG.dml", nvargs);

        System.out.println(Explain.explain(program));
        double time;
        try {
            time = CostEstimator.estimateExecutionTime(program, instanceMap.get("m5.xlarge"), null);
            System.out.println("Estimated time: " + time);
        } catch (CostEstimationException e) {
            throw new RuntimeException(e);
        }
        assertTrue(time > 0 && time != Double.MAX_VALUE);
    }

    @Test
    public void alsDSCostEstimationTestSingleNode() throws IOException {
        // test successful parsing
        Map<String, String> nvargs = new HashMap<>();
        nvargs.put("$X", "tests/X.csv");
        nvargs.put("$modelU", "tests/U.txt");
        nvargs.put("$modelV", "tests/V.txt");
        ResourceCompiler.setDriverConfigurations(GBtoBytes(16), 4);
        ResourceCompiler.setSingleNodeExecution();
        Program program = ResourceCompiler.compile("scripts/perftest/scripts/alsDS.dml", nvargs);

        System.out.println(Explain.explain(program));
        double time;
        try {
            time = CostEstimator.estimateExecutionTime(program, instanceMap.get("m5.xlarge"), null);
            System.out.println("Estimated time: " + time);
        } catch (CostEstimationException e) {
            throw new RuntimeException(e);
        }
        assertTrue(time > 0 && time != Double.MAX_VALUE);
    }

    @Test
    public void glmStatsSCostEstimationTestPassThroughSingleNode() throws IOException {
        // test successful parsing
        Map<String, String> nvargs = new HashMap<>();
        nvargs.put("$X", "tests/X.csv");
        nvargs.put("$Y", "tests/Y.csv");
        nvargs.put("$B", "tests/B.txt");
        ResourceCompiler.setDriverConfigurations(GBtoBytes(16), 4);
        ResourceCompiler.setSingleNodeExecution();
        Program program = ResourceCompiler.compile("scripts/perftest/scripts/GLM.dml", nvargs);

        System.out.println(Explain.explain(program));
        double time;
        try {
            time = CostEstimator.estimateExecutionTime(program, instanceMap.get("m5.xlarge"), null);
            System.out.println("Estimated time: " + time);
        } catch (CostEstimationException e) {
            throw new RuntimeException(e);
        }
        assertTrue(time > 0 && time != Double.MAX_VALUE);
    }

    @Test
    public void LinearRegCGCostEstimationTestSingleNode() throws IOException {
        // test successful parsing
        Map<String, String> nvargs = new HashMap<>();
        nvargs.put("$X", "tests/X.csv");
        nvargs.put("$Y", "tests/Y.csv");
        nvargs.put("$B", "tests/B.csv");
        ResourceCompiler.setDriverConfigurations(GBtoBytes(16), 4);
        ResourceCompiler.setSingleNodeExecution();
        Program program = ResourceCompiler.compile("scripts/perftest/scripts/LinearRegCG.dml", nvargs);

        System.out.println(Explain.explain(program.getDMLProg()));
        System.out.println(Explain.explain(program));
        double time;
        try {
            time = CostEstimator.estimateExecutionTime(program, instanceMap.get("m5.xlarge"), null);
            System.out.println("Estimated time: " + time);
        } catch (CostEstimationException e) {
            throw new RuntimeException(e);
        }
        assertTrue(time > 0 && time != Double.MAX_VALUE);
    }

    @Test
    public void LinearRegDSCostEstimationTestSingleNode() throws IOException {
        // test successful parsing
        Map<String, String> nvargs = new HashMap<>();
        nvargs.put("$X", "tests/X.csv");
        nvargs.put("$Y", "tests/Y.csv");
        nvargs.put("$B", "tests/B.csv");
        ResourceCompiler.setDriverConfigurations(GBtoBytes(4), 4);
        ResourceCompiler.setSingleNodeExecution();
        Program program = ResourceCompiler.compile("scripts/perftest/scripts/LinearRegDS.dml", nvargs);
        System.out.println(Explain.explain(program.getDMLProg()));
        System.out.println(Explain.explain(program));
        double time;
        try {
            time = CostEstimator.estimateExecutionTime(program, instanceMap.get("m5.xlarge"), null);
            System.out.println("Estimated time: " + time);
        } catch (CostEstimationException e) {
            throw new RuntimeException(e);
        }
        assertTrue(time > 0 && time != Double.MAX_VALUE);
    }

}
