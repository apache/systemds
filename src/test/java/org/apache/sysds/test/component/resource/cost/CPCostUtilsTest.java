package org.apache.sysds.test.component.resource.cost;

import org.apache.sysds.resource.cost.CPCostUtils;
import org.apache.sysds.resource.cost.VarStats;
import org.apache.sysds.runtime.instructions.cp.CPInstruction.CPType;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class CPCostUtilsTest {

    @Test
    public void testUnaryNotInstNFLOP() {
        long expectedValue = 1000 * 1000;
        testUnaryInstNFLOP("!", -1, -1, expectedValue);
    }

    @Test
    public void testUnaryIsnaInstNFLOP() {
        long expectedValue = 1000 * 1000;
        testUnaryInstNFLOP("isna", -1, -1, expectedValue);
    }

    @Test
    public void testUnaryIsnanInstNFLOP() {
        long expectedValue = 1000 * 1000;
        testUnaryInstNFLOP("isnan", -1, -1, expectedValue);
    }

    @Test
    public void testUnaryIsinfInstNFLOP() {
        long expectedValue = 1000 * 1000;
        testUnaryInstNFLOP("isinf", -1, -1, expectedValue);
    }

    @Test
    public void testUnaryCeilInstNFLOP() {
        long expectedValue = 1000 * 1000;
        testUnaryInstNFLOP("ceil", -1, -1, expectedValue);
    }

    @Test
    public void testUnaryFloorInstNFLOP() {
        long expectedValue = 1000 * 1000;
        testUnaryInstNFLOP("floor", -1, -1, expectedValue);
    }

    @Test
    public void testAbsInstNFLOPDefaultSparsity() {
        long expectedValue = 1000 * 1000;
        testUnaryInstNFLOP("abs", -1, -1, expectedValue);
    }

    @Test
    public void testAbsInstNFLOPSparse() {
        long expectedValue = (long) (0.5 * 1000 * 1000);
        testUnaryInstNFLOP("abs", 0.5, 0.5, expectedValue);
    }

    @Test
    public void testRoundInstNFLOPDefaultSparsity() {
        long expectedValue = 1000 * 1000;
        testUnaryInstNFLOP("round", -1, -1, expectedValue);
    }

    @Test
    public void testRoundInstNFLOPSparse() {
        long expectedValue = (long) (0.5 * 1000 * 1000);
        testUnaryInstNFLOP("round", 0.5, 0.5, expectedValue);
    }

    @Test
    public void testSignInstNFLOPDefaultSparsity() {
        long expectedValue = 1000 * 1000;
        testUnaryInstNFLOP("sign", -1, -1, expectedValue);
    }

    @Test
    public void testSignInstNFLOPSparse() {
        long expectedValue = (long) (0.5 * 1000 * 1000);
        testUnaryInstNFLOP("sign", 0.5, 0.5, expectedValue);
    }

    @Test
    public void testSpropInstNFLOPDefaultSparsity() {
        long expectedValue = 2 * 1000 * 1000;
        testUnaryInstNFLOP("sprop", -1, -1, expectedValue);
    }

    @Test
    public void testSpropInstNFLOPSparse() {
        long expectedValue = (long) (2 * 0.5 * 1000 * 1000);
        testUnaryInstNFLOP("sprop", 0.5, 0.5, expectedValue);
    }

    @Test
    public void testSqrtInstNFLOPDefaultSparsity() {
        long expectedValue = 2 * 1000 * 1000;
        testUnaryInstNFLOP("sqrt", -1, -1, expectedValue);
    }

    @Test
    public void testSqrtInstNFLOPSparse() {
        long expectedValue = (long) (2 * 0.5 * 1000 * 1000);
        testUnaryInstNFLOP("sqrt", 0.5, 0.5, expectedValue);
    }

    @Test
    public void testExpInstNFLOPDefaultSparsity() {
        long expectedValue = 18 * 1000 * 1000;
        testUnaryInstNFLOP("exp", -1, -1, expectedValue);
    }

    @Test
    public void testExpInstNFLOPSparse() {
        long expectedValue = (long) (18 * 0.5 * 1000 * 1000);
        testUnaryInstNFLOP("exp", 0.5, 0.5, expectedValue);
    }

    @Test
    public void testSigmoidInstNFLOPDefaultSparsity() {
        long expectedValue = 21 * 1000 * 1000;
        testUnaryInstNFLOP("sigmoid", -1, -1, expectedValue);
    }

    @Test
    public void testSigmoidInstNFLOPSparse() {
        long expectedValue = (long) (21 * 0.5 * 1000 * 1000);
        testUnaryInstNFLOP("sigmoid", 0.5, 0.5, expectedValue);
    }

    @Test
    public void testPlogpInstNFLOPDefaultSparsity() {
        long expectedValue = 32 * 1000 * 1000;
        testUnaryInstNFLOP("plogp", -1, -1, expectedValue);
    }

    @Test
    public void testPlogpInstNFLOPSparse() {
        long expectedValue = (long) (32 * 0.5 * 1000 * 1000);
        testUnaryInstNFLOP("plogp", 0.5, 0.5, expectedValue);
    }

    @Test
    public void testPrintInstNFLOP() {
        long expectedValue = 1000 * 1000;
        testUnaryInstNFLOP("print", -1, -1, expectedValue);
    }

    @Test
    public void testAssertInstNFLOP() {
        long expectedValue = 1000 * 1000;
        testUnaryInstNFLOP("assert", -1, -1, expectedValue);
    }

    @Test
    public void testSinInstNFLOPDefaultSparsity() {
        long expectedValue = 18 * 1000 * 1000;
        testUnaryInstNFLOP("sin", -1, -1, expectedValue);
    }

    @Test
    public void testSinInstNFLOPSparse() {
        long expectedValue = (long) (18 * 0.5 * 1000 * 1000);
        testUnaryInstNFLOP("sin", 0.5, 0.5, expectedValue);
    }

    @Test
    public void testCosInstNFLOPDefaultSparsity() {
        long expectedValue = 22 * 1000 * 1000;
        testUnaryInstNFLOP("cos", -1, -1, expectedValue);
    }

    @Test
    public void testCosInstNFLOPSparse() {
        long expectedValue = (long) (22 * 0.5 * 1000 * 1000);
        testUnaryInstNFLOP("cos", 0.5, 0.5, expectedValue);
    }

    @Test
    public void testTanInstNFLOPDefaultSparsity() {
        long expectedValue = 42 * 1000 * 1000;
        testUnaryInstNFLOP("tan", -1, -1, expectedValue);
    }

    @Test
    public void testTanInstNFLOPSparse() {
        long expectedValue = (long) (42 * 0.5 * 1000 * 1000);
        testUnaryInstNFLOP("tan", 0.5, 0.5, expectedValue);
    }

    @Test
    public void testAsinInstNFLOP() {
        long expectedValue = 93 * 1000 * 1000;
        testUnaryInstNFLOP("asin", -1, -1, expectedValue);
    }

    @Test
    public void testSinhInstNFLOP() {
        long expectedValue = 93 * 1000 * 1000;
        testUnaryInstNFLOP("sinh", -1, -1, expectedValue);
    }

    @Test
    public void testAcosInstNFLOP() {
        long expectedValue = 103 * 1000 * 1000;
        testUnaryInstNFLOP("acos", -1, -1, expectedValue);
    }

    @Test
    public void testCoshInstNFLOP() {
        long expectedValue = 103 * 1000 * 1000;
        testUnaryInstNFLOP("cosh", -1, -1, expectedValue);
    }

    @Test
    public void testAtanInstNFLOP() {
        long expectedValue = 40 * 1000 * 1000;
        testUnaryInstNFLOP("atan", -1, -1, expectedValue);
    }

    @Test
    public void testTanhInstNFLOP() {
        long expectedValue = 40 * 1000 * 1000;
        testUnaryInstNFLOP("tanh", -1, -1, expectedValue);
    }

    @Test
    public void testUcumkPlusInstNFLOPDefaultSparsity() {
        long expectedValue = 1000 * 1000;
        testUnaryInstNFLOP("ucumk+", -1, -1, expectedValue);
    }

    @Test
    public void testUcumkPlusInstNFLOPSparse() {
        long expectedValue = (long) (0.5 * 1000 * 1000);
        testUnaryInstNFLOP("ucumk+", 0.5, 0.5, expectedValue);
    }

    @Test
    public void testUcumMinInstNFLOPDefaultSparsity() {
        long expectedValue = 1000 * 1000;
        testUnaryInstNFLOP("ucummin", -1, -1, expectedValue);
    }

    @Test
    public void testUcumMinInstNFLOPSparse() {
        long expectedValue = (long) (0.5 * 1000 * 1000);
        testUnaryInstNFLOP("ucummin", 0.5, 0.5, expectedValue);
    }

    @Test
    public void testUcumMaxInstNFLOPDefaultSparsity() {
        long expectedValue = 1000 * 1000;
        testUnaryInstNFLOP("ucummax", -1, -1, expectedValue);
    }

    @Test
    public void testUcumMaxInstNFLOPSparse() {
        long expectedValue = (long) (0.5 * 1000 * 1000);
        testUnaryInstNFLOP("ucummax", 0.5, 0.5, expectedValue);
    }

    @Test
    public void testUcumMultInstNFLOPDefaultSparsity() {
        long expectedValue = 1000 * 1000;
        testUnaryInstNFLOP("ucum*", -1, -1, expectedValue);
    }

    @Test
    public void testUcumMultInstNFLOPSparse() {
        long expectedValue = (long) (0.5 * 1000 * 1000);
        testUnaryInstNFLOP("ucum*", 0.5, 0.5, expectedValue);
    }

    @Test
    public void testUcumkPlusMultInstNFLOPDefaultSparsity() {
        long expectedValue = 2 * 1000 * 1000;
        testUnaryInstNFLOP("ucumk+*", -1, -1, expectedValue);
    }

    @Test
    public void testUcumkPlusMultInstNFLOPSparse() {
        long expectedValue = (long) (2 * 0.5 * 1000 * 1000);
        testUnaryInstNFLOP("ucumk+*", 0.5, 0.5, expectedValue);
    }

    @Test
    public void testStopInstNFLOP() {
        long expectedValue = 0;
        testUnaryInstNFLOP("stop", -1, -1, expectedValue);
    }

    @Test
    public void testTypeofInstNFLOP() {
        long expectedValue = 1000 * 1000;
        testUnaryInstNFLOP("typeof", -1, -1, expectedValue);
    }

    @Test
    public void testInverseInstNFLOPDefaultSparsity() {
        long expectedValue = (long) ((4.0 / 3.0) * (1000 * 1000) * (1000 * 1000) * (1000 * 1000));
        testUnaryInstNFLOP("inverse", -1, -1, expectedValue);
    }

    @Test
    public void testInverseInstNFLOPSparse() {
        long expectedValue = (long) ((4.0 / 3.0) * (1000 * 1000) * (0.5 * 1000 * 1000) * (0.5 *1000 * 1000));
        testUnaryInstNFLOP("inverse", 0.5, 0.5, expectedValue);
    }

    @Test
    public void testCholeskyInstNFLOPDefaultSparsity() {
        long expectedValue = (long) ((1.0 / 3.0) * (1000 * 1000) * (1000 * 1000) * (1000 * 1000));
        testUnaryInstNFLOP("cholesky", -1, -1, expectedValue);
    }

    @Test
    public void testCholeskyInstNFLOPSparse() {
        long expectedValue = (long) ((1.0 / 3.0) * (1000 * 1000) * (0.5 * 1000 * 1000) * (0.5 *1000 * 1000));
        testUnaryInstNFLOP("cholesky", 0.5, 0.5, expectedValue);
    }

    @Test
    public void testLogInstNFLOP() {
        long expectedValue = 32 * 1000 * 1000;
        testBuiltinInstNFLOP("log", -1, expectedValue);
    }

    @Test
    public void testLogNzInstNFLOPDefaultSparsity() {
        long expectedValue = 32 * 1000 * 1000;
        testBuiltinInstNFLOP("log_nz", -1, expectedValue);
    }

    @Test
    public void testLogNzInstNFLOPSparse() {
        long expectedValue = (long) (32 * 0.5 * 1000 * 1000);
        testBuiltinInstNFLOP("log_nz", 0.5, expectedValue);
    }

    @Test
    public void testNrowInstNFLOP() {
        long expectedValue = 10L;
        testAggregateUnaryInstNFLOP("nrow", expectedValue);
    }

    @Test
    public void testNcolInstNFLOP() {
        long expectedValue = 10L;
        testAggregateUnaryInstNFLOP("ncol", expectedValue);
    }

    @Test
    public void testLengthInstNFLOP() {
        long expectedValue = 10L;
        testAggregateUnaryInstNFLOP("length", expectedValue);
    }

    @Test
    public void testExistsInstNFLOP() {
        long expectedValue = 10L;
        testAggregateUnaryInstNFLOP("exists", expectedValue);
    }

    @Test
    public void testLineageInstNFLOP() {
        long expectedValue = 10L;
        testAggregateUnaryInstNFLOP("lineage", expectedValue);
    }

    @Test
    public void testUakInstNFLOP() {
        long expectedValue = 4 * 1000 * 1000;
        testAggregateUnaryInstNFLOP("uak+", expectedValue);
    }

    @Test
    public void testUarkInstNFLOP() {
        long expectedValue = 4L * 2000 * 2000;
        testAggregateUnaryRowInstNFLOP("uark+", -1, expectedValue);
        testAggregateUnaryRowInstNFLOP("uark+", 0.5, expectedValue);
    }

    @Test
    public void testUackInstNFLOP() {
        long expectedValue = 4L * 3000 * 3000;
        testAggregateUnaryColInstNFLOP("uack+", -1, expectedValue);
        testAggregateUnaryColInstNFLOP("uack+", 0.5, expectedValue);
    }

    @Test
    public void testUasqkInstNFLOP() {
        long expectedValue = 5L * 1000 * 1000;
        testAggregateUnaryInstNFLOP("uasqk+", expectedValue);
    }

    @Test
    public void testUarsqkInstNFLOP() {
        long expectedValue = 5L * 2000 * 2000;
        testAggregateUnaryRowInstNFLOP("uarsqk+", -1, expectedValue);
        testAggregateUnaryRowInstNFLOP("uarsqk+", 0.5, expectedValue);
    }

    @Test
    public void testUacsqkInstNFLOP() {
        long expectedValue = 5L * 3000 * 3000;
        testAggregateUnaryColInstNFLOP("uacsqk+", -1, expectedValue);
        testAggregateUnaryColInstNFLOP("uacsqk+", 0.5, expectedValue);
    }

    @Test
    public void testUameanInstNFLOP() {
        long expectedValue = 7L * 1000 * 1000;
        testAggregateUnaryInstNFLOP("uamean", expectedValue);
    }

    @Test
    public void testUarmeanInstNFLOP() {
        long expectedValue = 7L * 2000 * 2000;
        testAggregateUnaryRowInstNFLOP("uarmean", -1, expectedValue);
        testAggregateUnaryRowInstNFLOP("uarmean", 0.5, expectedValue);
    }

    @Test
    public void testUacmeanInstNFLOP() {
        long expectedValue = 7L * 3000 * 3000;
        testAggregateUnaryColInstNFLOP("uacmean", -1, expectedValue);
        testAggregateUnaryColInstNFLOP("uacmean", 0.5, expectedValue);
    }

    @Test
    public void testUavarInstNFLOP() {
        long expectedValue = 14L * 1000 * 1000;
        testAggregateUnaryInstNFLOP("uavar", expectedValue);
    }

    @Test
    public void testUarvarInstNFLOP() {
        long expectedValue = 14L * 2000 * 2000;
        testAggregateUnaryRowInstNFLOP("uarvar", -1, expectedValue);
        testAggregateUnaryRowInstNFLOP("uarvar", 0.5, expectedValue);
    }

    @Test
    public void testUacvarInstNFLOP() {
        long expectedValue = 14L * 3000 * 3000;
        testAggregateUnaryColInstNFLOP("uacvar", -1, expectedValue);
        testAggregateUnaryColInstNFLOP("uacvar", 0.5, expectedValue);
    }

    @Test
    public void testUamaxInstNFLOP() {
        long expectedValue = 1000 * 1000;
        testAggregateUnaryInstNFLOP("uamax", expectedValue);
    }

    @Test
    public void testUarmaxInstNFLOP() {
        long expectedValue = 2000 * 2000;
        testAggregateUnaryRowInstNFLOP("uarmax", -1, expectedValue);
        testAggregateUnaryRowInstNFLOP("uarmax", 0.5, expectedValue);
    }

    @Test
    public void testUarimaxInstNFLOP() {
        long expectedValue = 2000 * 2000;
        testAggregateUnaryRowInstNFLOP("uarimax", -1, expectedValue);
        testAggregateUnaryRowInstNFLOP("uarimax", 0.5, expectedValue);
    }

    @Test
    public void testUacmaxInstNFLOP() {
        long expectedValue = 3000 * 3000;
        testAggregateUnaryColInstNFLOP("uacmax", -1, expectedValue);
        testAggregateUnaryColInstNFLOP("uacmax", 0.5, expectedValue);
    }

    @Test
    public void testUaminInstNFLOP() {
        long expectedValue = 1000 * 1000;
        testAggregateUnaryInstNFLOP("uamin", expectedValue);
    }

    @Test
    public void testUarminInstNFLOP() {
        long expectedValue = 2000 * 2000;
        testAggregateUnaryRowInstNFLOP("uarmin", -1, expectedValue);
        testAggregateUnaryRowInstNFLOP("uarmin", 0.5, expectedValue);
    }

    @Test
    public void testUariminInstNFLOP() {
        long expectedValue = 2000 * 2000;
        testAggregateUnaryRowInstNFLOP("uarimin", -1, expectedValue);
        testAggregateUnaryRowInstNFLOP("uarimin", 0.5, expectedValue);
    }

    @Test
    public void testUacminInstNFLOP() {
        long expectedValue = 3000 * 3000;
        testAggregateUnaryColInstNFLOP("uacmin", -1, expectedValue);
        testAggregateUnaryColInstNFLOP("uacmin", 0.5, expectedValue);
    }

    // HELPERS

    private void testUnaryInstNFLOP(String opcode, double sparsityIn, double sparsityOut, long expectedNFLOP) {
        long nnzIn = sparsityIn < 0? -1 : (long) (sparsityIn * 1000 * 1000);
        VarStats input = generateVarStatsMatrix("_mVar1", 1000, 1000, nnzIn);
        long nnzOut = sparsityOut < 0? -1 : (long) (sparsityOut * 1000 * 1000);
        VarStats output = generateVarStatsMatrix("_mVar2", 1000, 1000, nnzOut);

        long result = CPCostUtils.getInstNFLOP(CPType.Unary, opcode, output, input);
        assertEquals(expectedNFLOP, result);
    }

    private void testBuiltinInstNFLOP(String opcode, double sparsityIn, long expectedNFLOP) {
        long nnz = sparsityIn < 0? -1 : (long) (sparsityIn * 1000 * 1000);
        VarStats input = generateVarStatsMatrix("_mVar1", 1000, 1000, nnz);
        VarStats output = generateVarStatsMatrix("_mVar2", 1000, 1000, -1);

        long result = CPCostUtils.getInstNFLOP(CPType.Unary, opcode, output, input);
        assertEquals(expectedNFLOP, result);
    }

    private void testAggregateUnaryInstNFLOP(String opcode, long expectedNFLOP) {
        VarStats input = generateVarStatsMatrix("_mVar1", 1000, 1000, -1);
        VarStats output = generateVarStatsScalarLiteral("_Var2");

        long result = CPCostUtils.getInstNFLOP(CPType.AggregateUnary, opcode, output, input);
        assertEquals(expectedNFLOP, result);
    }

    private void testAggregateUnaryRowInstNFLOP(String opcode, double sparsityOut, long expectedNFLOP) {
        VarStats input = generateVarStatsMatrix("_mVar1", 2000, 1000, -1);
        long nnzOut = sparsityOut < 0? -1 : (long) (sparsityOut * 2000);
        VarStats output = generateVarStatsMatrix("_mVar2", 2000, 1, nnzOut);

        long result = CPCostUtils.getInstNFLOP(CPType.AggregateUnary, opcode, output, input);
        assertEquals(expectedNFLOP, result);
    }

    private void testAggregateUnaryColInstNFLOP(String opcode, double sparsityOut, long expectedNFLOP) {
        VarStats input = generateVarStatsMatrix("_mVar1", 1000, 3000, -1);
        long nnzOut = sparsityOut < 0? -1 : (long) (sparsityOut * 3000);
        VarStats output = generateVarStatsMatrix("_mVar2", 1, 3000, nnzOut);

        long result = CPCostUtils.getInstNFLOP(CPType.AggregateUnary, opcode, output, input);
        assertEquals(expectedNFLOP, result);
    }

    private VarStats generateVarStatsMatrix(String name, long rows, long cols, long nnz) {
        MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, nnz);
        return new VarStats(name, mc);
    }

    private VarStats generateVarStatsScalarLiteral(String nameOrValue) {
        return new VarStats(nameOrValue, null);
    }
}
