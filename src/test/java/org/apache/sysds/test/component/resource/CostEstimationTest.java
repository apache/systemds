package org.apache.sysds.test.component.resource;

import org.apache.sysds.resource.cost.CostEstimationException;
import org.apache.sysds.resource.cost.CostEstimator;
import org.apache.sysds.resource.cost.VarStats;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.CPInstruction;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.test.component.compress.TestBase;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;

public class CostEstimationTest {

    private CostEstimator estimator;

    @Before
    public void setup() {
        estimator = new CostEstimator();
    }

    @Test
    public void createvarMatrixVariableCPInstructionTest() {
        String instDefinition = "CP°createvar°testVar°testOutputFile°false°MATRIX°binary°100°100°1000°10000°COPY";
        VariableCPInstruction inst = VariableCPInstruction.parseInstruction(instDefinition);
        testGetTimeEstimateInst(estimator, null, inst, 0);
        // test the proper maintainCPInstVariableStatistics functionality
        VarStats actualStats = estimator.getStats("testVar");
        Assert.assertNotNull(actualStats);
        Assert.assertEquals(10000, actualStats.getCells());
    }

    @Test
    public void createvarFrameVariableCPInstructionTest() {
        String instDefinition = "CP°createvar°testVar°testOutputFile°false°FRAME°binary°100°100°1000°10000°COPY";
        VariableCPInstruction inst = VariableCPInstruction.parseInstruction(instDefinition);
        testGetTimeEstimateInst(estimator, null, inst, 0);
        // test the proper maintainCPInstVariableStatistics functionality
        VarStats actualStats = estimator.getStats("testVar");
        Assert.assertNotNull(actualStats);
        Assert.assertEquals(10000, actualStats.getCells());
    }

    @Test
    public void createvarInvalidVariableCPInstructionTest() {
        String instDefinition = "CP°createvar°testVar°testOutputFile°false°TENSOR°binary°100°100°1000°10000°COPY";
        VariableCPInstruction inst = VariableCPInstruction.parseInstruction(instDefinition);
        try {
            testGetTimeEstimateInst(estimator, null, inst, 0);
            Assert.fail("Tensor is not supported by the cost estimator");
        } catch (RuntimeException e) {
            // needed catch block to assert that RuntimeException has been thrown
        }
    }

    @Test
    public void cpvarVariableCPInstructionTest() {
        String instDefinition = "CP°createvar°testVar°testOutputFile°false°FRAME°binary°100°100°1000°10000°COPY";
        VariableCPInstruction inst = VariableCPInstruction.parseInstruction(instDefinition);
        testGetTimeEstimateInst(estimator, null, inst, 0);
        // test the proper maintainCPInstVariableStatistics functionality
        VarStats actualStats = estimator.getStats("testVar");
        Assert.assertNotNull(actualStats);
        Assert.assertEquals(10000, actualStats.getCells());
    }

    public void LinearRegCGCostEstimationTest() {

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
            actualCost = estimator.getTimeEstimateInst(null, targetInstruction);
        } catch (CostEstimationException e) {
            Assert.fail("Catching CostEstimationException is not expected behavior");
        }
        Assert.assertEquals(expectedCost, actualCost, 0.0);
    }
}
