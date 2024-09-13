package org.apache.sysds.test.component.resource;

import org.apache.sysds.resource.CloudInstance;
import org.apache.sysds.resource.ResourceCompiler;
import org.apache.sysds.resource.cost.CostEstimationException;
import org.apache.sysds.resource.cost.CostEstimator;
import org.apache.sysds.resource.cost.RDDStats;
import org.apache.sysds.resource.cost.VarStats;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.runtime.instructions.spark.BinarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.RandSPInstruction;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.utils.Explain;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.apache.sysds.resource.CloudUtils.GBtoBytes;
import static org.apache.sysds.test.component.resource.TestingUtils.getSimpleCloudInstanceMap;

public class CostEstimationTest {
    private static final HashMap<String, CloudInstance> instanceMap = getSimpleCloudInstanceMap();

    private CostEstimator estimator;

    @Before
    public void setup() {
        ResourceCompiler.setDriverConfigurations(GBtoBytes(8), 4);
        ResourceCompiler.setExecutorConfigurations(4, GBtoBytes(8), 4);
        estimator = new CostEstimator(new Program(), instanceMap.get("m5.xlarge"), instanceMap.get("m5.xlarge"));
    }

    @Test
    public void createvarMatrixVariableCPInstructionTest() {
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
    public void createvarFrameVariableCPInstructionTest() {
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
    public void createvarInvalidVariableCPInstructionTest() {
        String instDefinition = "CP°createvar°testVar°testOutputFile°false°TENSOR°binary°100°100°1000°10000°copy";
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
    public void randSPInstructionTest() throws CostEstimationException {
        HashMap<String, VarStats> dummyStats = new HashMap<>();
        dummyStats.put("matrixVar", generateStatsWithRdd("matrixVar", 1000000, 1000000, -1));
        dummyStats.put("outputVar", generateStats("outputVar", 1000000, 1000000, -1));
        estimator.putStats(dummyStats);
        String instDefinition = "SPARK°+°scalarVar·SCALAR·FP64·false°matrixVar·MATRIX·FP64°outputVar·MATRIX·FP64";
        BinarySPInstruction inst = BinarySPInstruction.parseInstruction(instDefinition);
        estimator.maintainStats(inst);
        estimator.parseSPInst(inst);
        double time = estimator.getStats("outputVar").getRddStats().getCost();
        System.out.println("Rand time="+time);
    }

    @Test
    public void plusBinaryMatrixMatrixSpInstructionTest() throws CostEstimationException {
        String instDefinition1 = "CP°createvar°matrixVar°testOutputFile°false°MATRIX°binary°1000000°1000000°1000°-1°copy";
        VariableCPInstruction inst1 = VariableCPInstruction.parseInstruction(instDefinition1);
        estimator.maintainStats(inst1);
        String instDefinition2 = "CP°createvar°outputVar°testOutputFile°false°MATRIX°binary°1000000°1000000°1000°-1°copy";
        VariableCPInstruction inst2 = VariableCPInstruction.parseInstruction(instDefinition2);
        estimator.maintainStats(inst2);
        String instDefinition3 = "SPARK°rand°1000000·SCALAR·INT64·true°1000000·SCALAR·INT64·true°1000°1°1°1.0°-1°null°uniform°1.0°matrixVar·MATRIX·FP64";
        RandSPInstruction inst3 = RandSPInstruction.parseInstruction(instDefinition3);
        estimator.maintainStats(inst3);
        estimator.parseSPInst(inst3);
        double time1 = estimator.getStats("matrixVar").getRddStats().getCost();
        String instDefinition4 = "SPARK°+°scalarVar·SCALAR·FP64·false°matrixVar·MATRIX·FP64°outputVar·MATRIX·FP64";
        BinarySPInstruction inst4 = BinarySPInstruction.parseInstruction(instDefinition4);
        estimator.maintainStats(inst4);
        estimator.parseSPInst(inst4);
        double time2 = estimator.getStats("outputVar").getRddStats().getCost();
        System.out.println("Rand time="+time1+"; sum time="+time2);
    }

    @Test
    public void dummyTest() throws IOException {
        Map<String, String> nvargs = new HashMap<>();
        ResourceCompiler.setDriverConfigurations(GBtoBytes(1), 4);
        ResourceCompiler.setExecutorConfigurations(2, GBtoBytes(4), 2);
        Program program = ResourceCompiler.compile("scripts/perftest/scripts/test.dml", nvargs);

        System.out.println(Explain.explain(program));
        try {
            CostEstimator.estimateExecutionTime(program, instanceMap.get("m5.xlarge"), instanceMap.get("m5.xlarge"));
        } catch (CostEstimationException e) {
            throw new RuntimeException(e);
        }
    }

    private VarStats generateStats(String name, long m, long n, long nnz) {
        MatrixCharacteristics mc = new MatrixCharacteristics(m, n, nnz);
        return new VarStats(name, mc);
    }

    private VarStats generateStatsWithRdd(String name, long m, long n, long nnz) {
        MatrixCharacteristics mc = new MatrixCharacteristics(m, n, nnz);
        VarStats stats = new VarStats(name, mc);
        RDDStats rddStats = new RDDStats(stats);
        stats.setRddStats(rddStats);
        return stats;
    }

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
