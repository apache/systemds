package org.apache.sysds.performance.matrix;

import org.apache.sysds.runtime.data.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.MemoryEstimates;
import org.junit.Test;

public class MatrixStorage extends AutomatedTestBase {

    private static final int resolution = 18;
    private static final float resolutionDivisor = 2f;
    private static final float maxSparsity = .4f;
    private static final float dimTestSparsity = .1f;

    static float[] sparsityProvider() {
        float[] sparsities = new float[resolution];
        float currentValue = maxSparsity;

        for (int i = 0; i < resolution; i++) {
            sparsities[i] = currentValue;
            currentValue /= resolutionDivisor;
        }

        return sparsities;
    }

    static int[][] dimsProvider(int rl, int maxCl, int minCl, int resolution) {
        int[][] dims = new int[2][resolution];
        for (int i = 0; i < resolution; i++) {
            dims[0][i] = rl;
            dims[1][i] = (int)(minCl + i * ((maxCl-minCl)/((float)resolution)));
        }

        return dims;
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

    static String printAsPythonList(int[] list) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");

        for (long el : list)
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

    /*@Test
    public void testDense() {
        testSparseFormat(null, 1024, 1024);
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
    }*/

    @Test
    public void testChangingDimsDense() {
        testChangingDims(null, dimTestSparsity, 1024, 10, 3000, 30);
    }

    @Test
    public void testChangingDimsMCSR() {
        testChangingDims(SparseBlock.Type.MCSR, dimTestSparsity, 1024, 10, 3000, 30);
    }

    @Test
    public void testChangingDimsCSR() {
        testChangingDims(SparseBlock.Type.CSR, dimTestSparsity, 1024, 10, 3000, 30);
    }

    @Test
    public void testChangingDimsCOO() {
        testChangingDims(SparseBlock.Type.COO, dimTestSparsity, 1024, 10, 3000, 30);
    }

    @Test
    public void testChangingDimsDCSR() {
        testChangingDims(SparseBlock.Type.DCSR, dimTestSparsity, 1024, 10, 3000, 30);
    }

    private void testSparseFormat(SparseBlock.Type btype, int rl, int cl) {
        float[] sparsities = MatrixStorage.sparsityProvider();
        long[] results = new long[sparsities.length];
        for (int sparsityIndex = 0; sparsityIndex < sparsities.length; sparsityIndex++)
            results[sparsityIndex] = evaluateMemoryConsumption(btype, sparsities[sparsityIndex], rl, cl);

        System.out.println("sparsities" + (btype == null ? "Dense" : btype.name()) + " = " + printAsPythonList(sparsities));
        System.out.println("memory" + (btype == null ? "Dense" : btype.name()) + " =  " + printAsPythonList(results));
    }

    private void testChangingDims(SparseBlock.Type btype, double sparsity, int rl, int minCl, int maxCl, int resolution) {
        int[][] dims = MatrixStorage.dimsProvider(rl, minCl, maxCl, resolution);
        long[] results = new long[resolution];
        for (int dimIndex = 0; dimIndex < resolution; dimIndex++)
            results[dimIndex] = evaluateMemoryConsumption(btype, sparsity, dims[0][dimIndex], dims[1][dimIndex]);

        System.out.println("dims" + (btype == null ? "Dense" : btype.name()) + " = " + printAsPythonList(dims[1]));
        System.out.println("dimMemory" + (btype == null ? "Dense" : btype.name()) + " =  " + printAsPythonList(results));
    }

    private long evaluateMemoryConsumption(SparseBlock.Type btype, double sparsity, int rl, int cl) {
        try
        {
            if (btype == null)
                return Math.min(Long.MAX_VALUE, (long)DenseBlockFP64.estimateMemory(rl, cl));

            double[][] A = getRandomMatrix(rl, cl, -10, 10, sparsity, 7654321);

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
