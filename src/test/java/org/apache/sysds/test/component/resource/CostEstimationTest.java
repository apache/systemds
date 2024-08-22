package org.apache.sysds.test.component.resource;

import org.apache.sysds.resource.ResourceCompiler;
import org.apache.sysds.resource.cost.CostEstimationException;
import org.apache.sysds.resource.cost.CostEstimator;
import org.apache.sysds.resource.cost.VarStats;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.utils.Explain;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class CostEstimationTest {

    private CostEstimator estimator;

    @Before
    public void setup() {
        estimator = new CostEstimator(new Program());
    }

    @Test
    public void createvarMatrixVariableCPInstructionTest() throws CostEstimationException {
        String instDefinition = "CP°createvar°testVar°testOutputFile°false°MATRIX°binary°100°100°1000°10000°COPY";
        VariableCPInstruction inst = VariableCPInstruction.parseInstruction(instDefinition);
        testGetTimeEstimateInst(estimator, null, inst, 0);
        // test the proper maintainCPInstVariableStatistics functionality
        estimator.maintainStats(inst);
        VarStats actualStats = estimator.getStats("testVar");
        Assert.assertNotNull(actualStats);
        Assert.assertEquals(10000, actualStats.getCells());
    }

    @Test
    public void createvarFrameVariableCPInstructionTest() throws CostEstimationException {
        String instDefinition = "CP°createvar°testVar°testOutputFile°false°FRAME°binary°100°100°1000°10000°COPY";
        VariableCPInstruction inst = VariableCPInstruction.parseInstruction(instDefinition);
        testGetTimeEstimateInst(estimator, null, inst, 0);
        // test the proper maintainCPInstVariableStatistics functionality
        estimator.maintainStats(inst);
        VarStats actualStats = estimator.getStats("testVar");
        Assert.assertNotNull(actualStats);
        Assert.assertEquals(10000, actualStats.getCells());
    }

    @Test
    public void createvarInvalidVariableCPInstructionTest() throws CostEstimationException {
        String instDefinition = "CP°createvar°testVar°testOutputFile°false°TENSOR°binary°100°100°1000°10000°COPY";
        VariableCPInstruction inst = VariableCPInstruction.parseInstruction(instDefinition);
        try {
            estimator.maintainStats(inst);
            testGetTimeEstimateInst(estimator, null, inst, 0);
            Assert.fail("Tensor is not supported by the cost estimator");
        } catch (RuntimeException e) {
            // needed catch block to assert that RuntimeException has been thrown
        }
    }

    @Test
    public void LinearRegCGCostEstimationTest() throws IOException {
        Map<String, String> nvargs = new HashMap<>();
        nvargs.put("$X", "tests/X.csv");
        nvargs.put("$Y", "tests/Y.csv");
        nvargs.put("$B", "tests/B.csv");
        Program program = ResourceCompiler.compile("scripts/perftest/scripts/LinearRegCG.dml", nvargs);
//        Program program = ResourceCompiler.compile("scripts/perftest/resource/all_ops.dml", nvargs);
        System.out.println(Explain.explain(program));
        try {
            CostEstimator.estimateExecutionTime(program);
        } catch (CostEstimationException e) {
            throw new RuntimeException(e);
        }
    }

    // Helper functions

    private static void testGetTimeEstimateInst(
            CostEstimator estimator,
            HashMap<String, VarStats> inputStats,
            Instruction targetInstruction,
            double expectedCost
    ) {
        if (inputStats != null)
            estimator.putStats(inputStats);
        double actualCost = -1d;
        try {
            actualCost = estimator.getTimeEstimateInst(targetInstruction);
        } catch (CostEstimationException e) {
            Assert.fail("Catching CostEstimationException is not expected behavior");
        }
        Assert.assertEquals(expectedCost, actualCost, 0.0);
    }
}
