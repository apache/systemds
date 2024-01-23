package org.apache.sysds.performance.matrix;

import org.apache.sysds.runtime.data.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class MatrixStorage extends AutomatedTestBase {

    private static final int resolution = 18;
    private static final float resolutionDivisor = 2f;
    private static final float maxSparsity = .4f;

    static float[] sparsityProvider() {
        float[] sparsities = new float[resolution];
        float currentValue = maxSparsity;

        for (int i = 0; i < resolution; i++) {
            sparsities[i] = currentValue;
            currentValue /= resolutionDivisor;
        }

        return sparsities;
    }

    static String printAsPythonList(float[] list) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");

        for (float el : list)
            sb.append(el + ",");

        if (list.length > 0)
            sb.deleteCharAt(sb.length() - 1);

        sb.append("]");
        return sb.toString();
    }

    static String printAsPythonList(long[] list) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");

        for (long el : list)
            sb.append(el + ",");

        if (list.length > 0)
            sb.deleteCharAt(sb.length() - 1);

        sb.append("]");
        return sb.toString();
    }

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
    }

    @Test
    public void testMCSR() {
        testSparseFormat(SparseBlock.Type.MCSR, 1024, 1024);
    }

    @Test
    public void testCSR() {
        testSparseFormat(SparseBlock.Type.CSR, 1024, 1024);
    }

    @Test
    public void testCOO() {
        testSparseFormat(SparseBlock.Type.COO, 1024, 1024);
    }

    @Test
    public void testDCSR() {
        testSparseFormat(SparseBlock.Type.DCSR, 1024, 1024);
    }

    private void testSparseFormat(SparseBlock.Type btype, int rl, int cl) {
        float[] sparsities = MatrixMulPerformance.sparsityProvider();
        long[] results = new long[sparsities.length];
        for (int sparsityIndex = 0; sparsityIndex < sparsities.length; sparsityIndex++)
            results[sparsityIndex] = evaluateMemoryConsumption(btype, sparsities[sparsityIndex], rl, cl);

        System.out.println("sparsities" + btype.name() + " = " + printAsPythonList(sparsities));
        System.out.println("memory" + btype.name() + " =  " + printAsPythonList(results));
    }

    private long evaluateMemoryConsumption(SparseBlock.Type btype, double sparsity, int rl, int cl) {
        try
        {
            double[][] A = getRandomMatrix(rl, cl, -10, 10, sparsity, 7654321);

            if (btype == null)
                return ((long)(A.length))*A[0].length*8;

            MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);

            if (!mbtmp.isInSparseFormat())
                mbtmp.denseToSparse(true);

            SparseBlock srtmp = mbtmp.getSparseBlock();
            switch (btype) {
                case MCSR:
                    SparseBlockMCSR mcsr = new SparseBlockMCSR(srtmp);
                    return mcsr.getExactSizeInMemory();
                case CSR:
                    SparseBlockCSR csr = new SparseBlockCSR(srtmp);
                    return csr.getExactSizeInMemory();
                case COO:
                    SparseBlockCOO coo = new SparseBlockCOO(srtmp);
                    return coo.getExactSizeInMemory();
                case DCSR:
                    SparseBlockDCSR dcsr = new SparseBlockDCSR(srtmp);
                    return dcsr.getExactSizeInMemory();
            }


        } catch(Exception ex) {
            ex.printStackTrace();
            throw new RuntimeException(ex);
        }
        throw new IllegalArgumentException();
    }
}
