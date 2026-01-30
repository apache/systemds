package org.apache.sysds.test.component.matrix.libMatrixReorg;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.data.DenseBlock;
import org.mockito.Mockito;
import java.util.Arrays;

public class PermuteTest {

    @Test
    public void testBasicPermute() {
        int[] shape = {2, 3, 4};
        MatrixBlock tensor = generateMatrixBlock(shape);
        
        Assert.assertEquals(24, tensor.getNumRows() * tensor.getNumColumns());
        
        double[] data = tensor.getDenseBlockValues();
        Assert.assertEquals(23.0, data[1 * 4 * 3 + 2 * 4 + 3], 0.001);
        Assert.assertEquals(0.0, data[0 * 4 * 3 + 0 * 4 + 0], 0.001);

        int[] permutation = {1, 0, 2};
        MatrixBlock outTensor = LibMatrixReorg.permute(tensor, shape, permutation); 

        double[] outData = outTensor.getDenseBlockValues();
        Assert.assertEquals(24, outData.length); 
        Assert.assertEquals(4.0, outData[8], 0.001);
        Assert.assertEquals(15.0, outData[7], 0.001);
    }

    @Test
    public void testPermute2D_Transpose() {
        int[] shape = {10, 5};
        int[] perm = {1, 0};
        
        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);
        
        verifyPermutation(in, out, shape, perm);
    }

    @Test
    public void testPermute3D_Simple() {
        int[] shape = {2, 3, 4};
        int[] perm = {1, 0, 2};

        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

        verifyPermutation(in, out, shape, perm);
    }

    @Test
    public void testPermute3D_Identity() {
        int[] shape = {5, 5, 5};
        int[] perm = {0, 1, 2};

        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

        verifyPermutation(in, out, shape, perm);
    }

    @Test
    public void testPermute4D_Reverse() {
        int[] shape = {2, 3, 4, 5};
        int[] perm = {3, 2, 1, 0};

        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

        verifyPermutation(in, out, shape, perm);
    }

    @Test
    public void testPermuteHighRank() {
        int[] shape = {2, 2, 2, 2, 2, 2};
        int[] perm = {5, 0, 4, 1, 3, 2};

        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

        verifyPermutation(in, out, shape, perm);
    }

    @Test
    public void testLargeBlockLogic_Mocked() {
        int[] shape = {10, 10, 10};
        int[] perm = {2, 0, 1};

        MatrixBlock in = generateMatrixBlock(shape);
        DenseBlock originalDB = in.getDenseBlock();
        DenseBlock spyDB = Mockito.spy(originalDB);
        Mockito.when(spyDB.numBlocks()).thenReturn(2);
        in.setDenseBlock(spyDB);

        MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);

        MatrixBlock originalIn = generateMatrixBlock(shape);
        verifyPermutation(originalIn, out, shape, perm);
    }

    @Test
    public void testLargeBlockLogic_Mocked_InputAndOutput() {
        int[] shape = {4, 4, 4};
        int[] perm = {2, 1, 0};
        
        MatrixBlock in = generateMatrixBlock(shape);
        DenseBlock spyIn = Mockito.spy(in.getDenseBlock());
        Mockito.when(spyIn.numBlocks()).thenReturn(5);
        in.setDenseBlock(spyIn);
        
        MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);
        
        MatrixBlock originalIn = generateMatrixBlock(shape);
        verifyPermutation(originalIn, out, shape, perm);
    }

    @Test
    public void testPermute3D_Parallel() {
        int[] shape = {100, 100, 100};
        int[] perm = {2, 0, 1};
        
        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock out = LibMatrixReorg.permute(in, shape, perm, -1);
        
        verifyPermutation(in, out, shape, perm);
    }

    @Test
    public void testPerformance_SingleVsMultiThreaded() {
        int size = 100; 
        int[] shape = {size, size, size};
        int[] perm = {2, 0, 1};
        
        MatrixBlock in = generateMatrixBlock(shape);
        
        long startSingle = System.nanoTime();
        MatrixBlock outSingle = LibMatrixReorg.permute(in, shape, perm, 1);
        long timeSingle = System.nanoTime() - startSingle;
        
        long startMulti = System.nanoTime();
        MatrixBlock outMulti = LibMatrixReorg.permute(in, shape, perm, -1);
        long timeMulti = System.nanoTime() - startMulti;
        
        verifyPermutation(in, outSingle, shape, perm);
        verifyPermutation(in, outMulti, shape, perm);
        
        System.out.println("Large Matrix (" + size + "x" + size + "x" + size + "):");
        System.out.println("Single-threaded: " + timeSingle / 1_000_000 + " ms");
        System.out.println("Multi-threaded: " + timeMulti / 1_000_000 + " ms");
        System.out.println("Speedup: " + String.format("%.2fx", (double)timeSingle / timeMulti));

        Assert.assertTrue("Multi-threaded should be faster for large matrices", timeMulti < timeSingle);
    }

    @Test
    public void testPerformance_LargeMatrix_SingleVsMulti() {
        int[] shape = {1, 10000, 10000};
        int[] perm = {0, 2, 1};
        
        MatrixBlock in = generateMatrixBlock(shape);
        
        long startSingle = System.nanoTime();
        MatrixBlock outSingle = LibMatrixReorg.permute(in, shape, perm, 1);
        long timeSingle = System.nanoTime() - startSingle;
        
        long startMulti = System.nanoTime();
        MatrixBlock outMulti = LibMatrixReorg.permute(in, shape, perm, -1);
        long timeMulti = System.nanoTime() - startMulti;
        
        System.out.println("Large Matrix (" + 1 + "x" + 10000 + "x" + 100000 + "):");
        System.out.println("Single-threaded: " + timeSingle / 1_000_000 + " ms");
        System.out.println("Multi-threaded: " + timeMulti / 1_000_000 + " ms");
        System.out.println("Speedup: " + String.format("%.2fx", (double)timeSingle / timeMulti));
        
        Assert.assertTrue("Multi-threaded should be faster for large matrices", timeMulti < timeSingle);
    }

    @Test
    public void testPerformance_PermuteVsNativeTranspose() {
        int size = 1000;
        MatrixBlock in = new MatrixBlock(size, size, false);
        in.allocateDenseBlock();
        double[] data = in.getDenseBlockValues();
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                data[i * size + j] = i * size + j;
            }
        }
        
        int[] shape = {size, size};
        int[] perm = {1, 0};
        
        long startPermute = System.nanoTime();
        MatrixBlock outPermute = LibMatrixReorg.permute(in, shape, perm, -1);
        long timePermute = System.nanoTime() - startPermute;
        
        long startTranspose = System.nanoTime();
        MatrixBlock outTranspose = LibMatrixReorg.transpose(in);
        long timeTranspose = System.nanoTime() - startTranspose;
        
        System.out.println("Transpose Performance (" + size + "x" + size + "):");
        System.out.println("Permute function: " + timePermute / 1_000_000 + " ms");
        System.out.println("Native transpose: " + timeTranspose / 1_000_000 + " ms");
        System.out.println("Ratio: " + String.format("%.2fx", (double)timePermute / timeTranspose));
        
        double[] permuteData = outPermute.getDenseBlockValues();
        
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double expected = in.get(j, i);
                double actual = permuteData[i * size + j];
                Assert.assertEquals("Mismatch at (" + i + "," + j + ")", expected, actual, 0.0001);
            }
        }
    }

    @Test
    public void testEdgeCase_SingleElement() {
        int[] shape = {1, 1, 1};
        int[] perm = {2, 1, 0};
        
        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);
        
        verifyPermutation(in, out, shape, perm);
    }

    @Test
    public void testEdgeCase_OneDimensionOne() {
        int[] shape = {5, 1, 10};
        int[] perm = {2, 0, 1};
        
        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);
        
        verifyPermutation(in, out, shape, perm);
    }

    @Test
    public void testEdgeCase_TwoDimensionsOne() {
        int[] shape = {1, 1, 100};
        int[] perm = {2, 1, 0};
        
        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);
        
        verifyPermutation(in, out, shape, perm);
    }

    @Test
    public void testConsecutivePermutations() {
        int[] shape = {3, 4, 5};
        int[] perm1 = {1, 0, 2};
        int[] perm2 = {2, 0, 1};
        
        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock temp = LibMatrixReorg.permute(in, shape, perm1);
        
        int[] tempShape = {shape[perm1[0]], shape[perm1[1]], shape[perm1[2]]};
        MatrixBlock out = LibMatrixReorg.permute(temp, tempShape, perm2);
        
        int[] finalShape = {tempShape[perm2[0]], tempShape[perm2[1]], tempShape[perm2[2]]};
        
        verifyPermutation(temp, out, tempShape, perm2);
    }

    @Test
    public void testDifferentThreadCounts() {
        int[] shape = {50, 50, 50};
        int[] perm = {2, 0, 1};
        
        MatrixBlock in = generateMatrixBlock(shape);
        
        MatrixBlock out1 = LibMatrixReorg.permute(in, shape, perm, 1);
        MatrixBlock out2 = LibMatrixReorg.permute(in, shape, perm, 2);
        MatrixBlock out4 = LibMatrixReorg.permute(in, shape, perm, 4);
        MatrixBlock out8 = LibMatrixReorg.permute(in, shape, perm, 8);
        
        double[] data1 = out1.getDenseBlockValues();
        double[] data2 = out2.getDenseBlockValues();
        double[] data4 = out4.getDenseBlockValues();
        double[] data8 = out8.getDenseBlockValues();
        
        for (int i = 0; i < data1.length; i++) {
            Assert.assertEquals(data1[i], data2[i], 0.0001);
            Assert.assertEquals(data1[i], data4[i], 0.0001);
            Assert.assertEquals(data1[i], data8[i], 0.0001);
        }
    }

    @Test
    public void testPermute_AllDimensionsCyclic() {
        int[] shape = {3, 4, 5, 2};
        int[] perm = {1, 2, 3, 0};
        
        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);
        
        verifyPermutation(in, out, shape, perm);
    }

    @Test
    public void testPermute_NonContiguousStrides() {
        int[] shape = {7, 11, 13};
        int[] perm = {2, 0, 1};
        
        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);
        
        verifyPermutation(in, out, shape, perm);
    }

    @Test
    public void testPermute_LargePrimeStrides() {
        int[] shape = {17, 19};
        int[] perm = {1, 0};
        
        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock out = LibMatrixReorg.permute(in, shape, perm);
        
        verifyPermutation(in, out, shape, perm);
    }

    private MatrixBlock generateMatrixBlock(int[] shape) {
        long len = 1;
        for (int d : shape) len *= d;
        
        MatrixBlock mb = new MatrixBlock(1, (int)len, false);
        mb.allocateDenseBlock();
        double[] data = mb.getDenseBlockValues();
        for (int i = 0; i < data.length; i++) {
            data[i] = (double) i;
        }
        return mb;
    }

    private void verifyPermutation(MatrixBlock in, MatrixBlock out, int[] inShape, int[] perm) {
        double[] inData = new double[(int)(in.getNumRows() * in.getNumColumns())];
        double[] outData = new double[(int)(out.getNumRows() * out.getNumColumns())];
        
        DenseBlock inDB = in.getDenseBlock();
        DenseBlock outDB = out.getDenseBlock();
        
        if (inDB != null) {
            int inBlockSize = inDB.blockSize();
            for (int i = 0; i < inDB.numBlocks(); i++) {
                double[] block = inDB.valuesAt(i);
                int offset = i * inBlockSize;
                int len = Math.min(inBlockSize, inData.length - offset);
                System.arraycopy(block, 0, inData, offset, len);
            }
        }
        
        if (outDB != null) {
            int outBlockSize = outDB.blockSize();
            for (int i = 0; i < outDB.numBlocks(); i++) {
                double[] block = outDB.valuesAt(i);
                int offset = i * outBlockSize;
                int len = Math.min(outBlockSize, outData.length - offset);
                System.arraycopy(block, 0, outData, offset, len);
            }
        }
        
        int rank = inShape.length;
        int[] outShape = new int[rank];
        for (int i = 0; i < rank; i++) 
            outShape[i] = inShape[perm[i]];

        long[] outStrides = getStrides(outShape);
        long[] inStrides = getStrides(inShape);

        long len = 1;
        for (int d : outShape) len *= d;

        for (long i = 0; i < len; i++) {
            int[] outCoords = new int[rank];
            long temp = i;
            for (int d = 0; d < rank; d++) {
                outCoords[d] = (int)(temp / outStrides[d]);
                temp = temp % outStrides[d];
            }

            int[] inCoords = new int[rank];
            for (int d = 0; d < rank; d++) {
                inCoords[perm[d]] = outCoords[d];
            }
            
            long inIndex = 0;
            for (int d = 0; d < rank; d++) {
                inIndex += inCoords[d] * inStrides[d];
            }
            
            double expectedValue = inData[(int)inIndex];
            double actualValue = outData[(int)i];
            
            if (Math.abs(expectedValue - actualValue) > 0.0001) {
                Assert.fail("Mismatch at linear output index " + i + 
                            ". Output coords " + Arrays.toString(outCoords) + 
                            ". Input coords " + Arrays.toString(inCoords) +
                            ". Expected " + expectedValue + " but got " + actualValue);
            }
        }
    }

    private long[] getStrides(int[] dims) {
        long[] strides = new long[dims.length];
        long stride = 1;
        for (int i = dims.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= dims[i];
        }
        return strides;
    }
}