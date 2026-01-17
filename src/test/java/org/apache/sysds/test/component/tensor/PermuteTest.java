package org.apache.sysds.test.component.tensor;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.data.DenseBlock;
//import org.apache.sysds.runtime.data.DenseBlockFactory;
import org.mockito.Mockito;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.List;
import java.util.ArrayList;


public class PermuteTest {

    @Test
    public void TestMatrixBlockPermute() {

        int[] shape = {2, 3, 4};

        MatrixBlock tensor = TensorUtils.createArangeMatrixBlock(shape);
        Assert.assertEquals(24, tensor.getNumRows() * tensor.getNumColumns());
        
        double[] data = tensor.getDenseBlockValues();
        Assert.assertEquals(23.0, data[1 * 4 * 3 + 2 * 4 + 3], 0.001);
        Assert.assertEquals( 0.0, data[0 * 4 * 3 + 0 * 4 + 0], 0.001);

        TensorUtils.printMatrixTensor(tensor, shape);

        int[] permutation = {1, 0, 2};
        
        MatrixBlock outTensor = PermuteIt.permute(tensor, shape, permutation); 
        int[] outShape = {3, 2, 4};

        TensorUtils.printMatrixTensor(outTensor, outShape); 

        double[] outData = outTensor.getDenseBlockValues();
        Assert.assertEquals(24, 1 * outTensor.getNumColumns());
        Assert.assertEquals(24, outData.length); 
        Assert.assertEquals(4.0, outData[8], 0.001);
        Assert.assertEquals(15.0, outData[7], 0.001);
    }

    @Test
    public void testPermute2D_Transpose() {
        int[] shape = {10, 5};
        int[] perm = {1, 0};
        
        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock out = PermuteIt.permute(in, shape, perm);
        
        verifyPermutation(in, out, shape, perm);
    }

    @Test
    public void testPermute3D_Simple() {
        int[] shape = {2, 3, 4};
        int[] perm = {1, 0, 2};

        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock out = PermuteIt.permute(in, shape, perm);

        verifyPermutation(in, out, shape, perm);
    }

    @Test
    public void testPermute3D_Identity() {
        int[] shape = {5, 5, 5};
        int[] perm = {0, 1, 2};

        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock out = PermuteIt.permute(in, shape, perm);

        verifyPermutation(in, out, shape, perm);
    }

    @Test
    public void testPermute4D_Reverse() {
        int[] shape = {2, 3, 4, 5};
        int[] perm = {3, 2, 1, 0};

        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock out = PermuteIt.permute(in, shape, perm);

        verifyPermutation(in, out, shape, perm);
    }

    @Test
    public void testPermuteHighRank() {
        int[] shape = {2, 2, 2, 2, 2, 2};
        int[] perm = {5, 0, 4, 1, 3, 2};

        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock out = PermuteIt.permute(in, shape, perm);

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

        MatrixBlock out = PermuteIt.permute(in, shape, perm);

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
        
        MatrixBlock out = PermuteIt.permute(in, shape, perm);
        
        MatrixBlock originalIn = generateMatrixBlock(shape);
        verifyPermutation(originalIn, out, shape, perm);
    }

    @Test
    public void testPermute3D_Parallel() {
        int[] shape = {100, 100, 100};
        int[] perm = {2, 0, 1};
        
        MatrixBlock in = generateMatrixBlock(shape);
        MatrixBlock out = PermuteIt.permute(in, shape, perm, -1);
        
        verifyPermutation(in, out, shape, perm);
    }


    private MatrixBlock generateMatrixBlock(int[] shape) {
        long len = 1;
        for (int d : shape) len *= d;
        
        MatrixBlock mb = new MatrixBlock(1, (int)len, false);
        mb.allocateDenseBlock();
        double[] data = mb.getDenseBlockValues();
        for(int i = 0; i < data.length; i++) {
            data[i] = (double)i;
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
        for(int i=0; i<rank; i++) outShape[i] = inShape[perm[i]];

        long[] outStrides = getStrides(outShape);
        long[] inStrides = getStrides(inShape);

        long len = 1;
        for (int d : inShape) len *= d;

        for(long i = 0; i < len; i++) {
            double actualValue = outData[(int)i];

            int[] currentCoords = new int[rank];
            long temp = i;
            for(int d = 0; d < rank; d++) {
                currentCoords[d] = (int)(temp / outStrides[d]);
                temp = temp % outStrides[d];
            }

            int[] inCoords = new int[rank];
            for(int d = 0; d < rank; d++) {
                inCoords[d] = currentCoords[perm[d]];
            }
            
            long inIndex = 0;
            for(int d = 0; d < rank; d++) {
                inIndex += inCoords[d] * inStrides[d];
            }
            
            double expectedValue = inData[(int)inIndex];
            
            if(Math.abs(expectedValue - actualValue) > 0.0001) {
                Assert.fail("Mismatch at linear output index " + i + 
                            ". Output coords " + Arrays.toString(currentCoords) + 
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


    public static class TensorUtils {

        public static MatrixBlock createArangeMatrixBlock(int[] shape) {
            long length = 1;
            for (int d : shape) length *= d;

            MatrixBlock mb = new MatrixBlock(1, (int)length, false);
            mb.allocateDenseBlock();
            
            double[] data = mb.getDenseBlockValues();
            for (int i = 0; i < data.length; i++) {
                data[i] = (double) i;
            }
            return mb;
        }

        public static void printMatrixTensor(MatrixBlock mb, int[] shape) {
            double[] data = mb.getDenseBlockValues();
            StringBuilder sb = new StringBuilder();
            sb.append("MatrixBlock-Tensor(").append(Arrays.toString(shape)).append("):\n");
            printRecursive(data, shape, 0, 0, sb, 0);
            System.out.println(sb.toString());
        }

        private static void printRecursive(double[] data, int[] shape, int dim, int offset, StringBuilder sb, int indent) {
            int stride = 1;
            for (int i = dim + 1; i < shape.length; i++) stride *= shape[i];

            for (int k = 0; k < indent; k++) sb.append(" ");
            sb.append("[");

            if (dim == shape.length - 1) {
                for (int i = 0; i < shape[dim]; i++) {
                    sb.append(String.format("%.1f", data[offset + i]));
                    if (i < shape[dim] - 1) sb.append(", ");
                }
                sb.append("]");
            } else {
                sb.append("\n");
                for (int i = 0; i < shape[dim]; i++) {
                    printRecursive(data, shape, dim + 1, offset + i * stride, sb, indent + 2);
                    if (i < shape[dim] - 1) {
                        sb.append(",");
                        sb.append("\n");
                        if (shape.length - dim > 2) sb.append("\n");
                    }
                }
                sb.append("\n");
                for (int k = 0; k < indent; k++) sb.append(" ");
                sb.append("]");
            }
        }
    }


    public static class PermuteIt {

        // blocking according to typical L2 cache sizes
        private static final int BLOCK_SIZE = 128;
        private static final int PAR_NUMCELL_THRESHOLD = 1024; //1024*1024 

        //Aus LibMatrixReorg
        static void transposeRow(double[] a, double[] c, int aix, int cix, int n2, int len) {
            final int bn = len % 8;
            for (int j = 0; j < bn; j++, aix++, cix += n2)
                c[cix] = a[aix];
            for (int j = bn; j < len; j += 8, aix += 8, cix += 8 * n2) {
                c[cix + 0 * n2] = a[aix + 0];
                c[cix + 1 * n2] = a[aix + 1];
                c[cix + 2 * n2] = a[aix + 2];
                c[cix + 3 * n2] = a[aix + 3];
                c[cix + 4 * n2] = a[aix + 4];
                c[cix + 5 * n2] = a[aix + 5];
                c[cix + 6 * n2] = a[aix + 6];
                c[cix + 7 * n2] = a[aix + 7];
            }
        }
        
        private static long[] getStrides(int[] dims) {
            long[] strides = new long[dims.length];
            long stride = 1;
            for (int i = dims.length - 1; i >= 0; i--) {
                strides[i] = stride;
                stride *= dims[i];
            }
            return strides;
        }

        public static MatrixBlock permute(MatrixBlock in, int[] inDims, int[] perm) {
            return permute(in, inDims, perm, 1);
        }

        public static MatrixBlock permute(MatrixBlock in, int[] inDims, int[] perm, int k) {
            int rank = inDims.length;
            
            //Early opt out 
            boolean isIdentity = true;
            for (int i = 0; i < rank; i++) {
                if (perm[i] != i) {
                    isIdentity = false;
                    break;
                }
            }
            if (isIdentity) {
                return new MatrixBlock(in);
            }

            int[] outDims = new int[rank];
            for (int i = 0; i < rank; i++) 
                outDims[i] = inDims[perm[i]];

            long length = 1;
            for (int d : outDims) length *= d;

            MatrixBlock out = new MatrixBlock(1, (int)length, false);
            out.allocateDenseBlock();

            DenseBlock inDB = in.getDenseBlock();
            DenseBlock outDB = out.getDenseBlock();

            long[] inStrides = getStrides(inDims);
            long[] outStrides = getStrides(outDims);
            
            long[] permutedStrides = new long[rank];
            for (int i = 0; i < rank; i++) {
                permutedStrides[i] = outStrides[perm[i]];
            }

            boolean useParallel = (k > 1 || k == -1) && length >= PAR_NUMCELL_THRESHOLD;
            int numThreads = k == -1 ? Runtime.getRuntime().availableProcessors() : k;

            if (inDB.numBlocks() == 1 && outDB.numBlocks() == 1) {
                double[] inData = inDB.valuesAt(0);
                double[] outData = outDB.valuesAt(0);
                
                if (useParallel && rank > 0) {
                    parallelPermuteSingleBlock(inData, outData, inDims, inStrides, 
                        permutedStrides, numThreads);
                } else {
                    recursivePermuteSingleBlock(inData, outData, inDims, inStrides, 
                        permutedStrides, 0, 0, 0);
                }
            } 
            else {
                if (useParallel && rank > 0) {
                    parallelPermuteMultiBlock(inDB, outDB, inDims, inStrides, 
                        permutedStrides, numThreads);
                } else {
                    recursivePermuteMultiBlock(inDB, outDB, inDims, inStrides, 
                        permutedStrides, 0, 0L, 0L);
                }
            }
            return out;
        }

        private static void recursivePermuteSingleBlock(
                double[] inData, double[] outData,
                int[] inDims, long[] inStrides, long[] permutedStrides,
                int dim, int inOffset, int outOffset) {

            if (dim == inDims.length - 1) {
                int len = inDims[dim];
                int outStride = (int) permutedStrides[dim];

                if (outStride == 1) 
                    System.arraycopy(inData, inOffset, outData, outOffset, len);
                else 
                    transposeRow(inData, outData, inOffset, outOffset, outStride, len);
                return;
            }

            int dimSize = inDims[dim];
            long inStep = inStrides[dim];
            long outStep = permutedStrides[dim];

            for (int bi = 0; bi < dimSize; bi += BLOCK_SIZE) {
                int bimin = Math.min(bi + BLOCK_SIZE, dimSize);
                for (int i = bi; i < bimin; i++) {
                    recursivePermuteSingleBlock(
                            inData, outData, inDims, inStrides, permutedStrides,
                            dim + 1,
                            inOffset + (int)(i * inStep),
                            outOffset + (int)(i * outStep)
                    );
                }
            }
        }

        private static void parallelPermuteSingleBlock(
                double[] inData, double[] outData,
                int[] inDims, long[] inStrides, long[] permutedStrides,
                int numThreads) {
            
            int dimSize = inDims[0];
            int tasksPerThread = Math.max(1, dimSize / numThreads);
            
            ExecutorService pool = Executors.newFixedThreadPool(numThreads);
            List<Future<?>> futures = new ArrayList<>();
            
            for (int t = 0; t < numThreads; t++) {
                final int start = t * tasksPerThread;
                final int end = (t == numThreads - 1) ? dimSize : (t + 1) * tasksPerThread;
                
                if (start >= dimSize) break;
                
                futures.add(pool.submit(() -> {
                    for (int i = start; i < end; i++) {
                        recursivePermuteSingleBlock(
                            inData, outData, inDims, inStrides, permutedStrides,
                            1,
                            (int)(i * inStrides[0]),
                            (int)(i * permutedStrides[0])
                        );
                    }
                }));
            }
            
            for (Future<?> f : futures) {
                try {
                    f.get();
                } catch (Exception e) {
                    throw new RuntimeException("Parallel permute failed", e);
                }
            }
            pool.shutdown();
        }

        private static void recursivePermuteMultiBlock(
            DenseBlock inDB, DenseBlock outDB,
            int[] inDims, long[] inStrides, long[] permutedStrides,
            int dim, long inOffset, long outOffset) {

            if (dim == inDims.length - 1) {
                int len = inDims[dim];
                long outStride = permutedStrides[dim];

                int inBlockSize = inDB.blockSize();
                int outBlockSize = outDB.blockSize();

                for (int i = 0; i < len; i++) {
                    long currentInAbs = inOffset + i * inStrides[dim];
                    long currentOutAbs = outOffset + i * outStride;
                    
                    int inBlockIdx = (int) (currentInAbs / inBlockSize);
                    int inRelIdx = (int) (currentInAbs % inBlockSize);
                    
                    int outBlockIdx = (int) (currentOutAbs / outBlockSize);
                    int outRelIdx = (int) (currentOutAbs % outBlockSize);
                    
                    double[] inArr = inDB.valuesAt(inBlockIdx);
                    double[] outArr = outDB.valuesAt(outBlockIdx);
                    
                    if (inArr != null && outArr != null && 
                        inRelIdx < inArr.length && outRelIdx < outArr.length) {
                        outArr[outRelIdx] = inArr[inRelIdx];
                    }
                }
                return;
            }

            int dimSize = inDims[dim];
            long inStep = inStrides[dim];
            long outStep = permutedStrides[dim];

            for (int bi = 0; bi < dimSize; bi += BLOCK_SIZE) {
                int bimin = Math.min(bi + BLOCK_SIZE, dimSize);
                for (int i = bi; i < bimin; i++) {
                    recursivePermuteMultiBlock(
                            inDB, outDB, inDims, inStrides, permutedStrides,
                            dim + 1,
                            inOffset + i * inStep,
                            outOffset + i * outStep
                    );
                }
            }
        }

        private static void parallelPermuteMultiBlock(
                DenseBlock inDB, DenseBlock outDB,
                int[] inDims, long[] inStrides, long[] permutedStrides,
                int numThreads) {
            
            int dimSize = inDims[0];
            int tasksPerThread = Math.max(1, dimSize / numThreads);
            
            ExecutorService pool = Executors.newFixedThreadPool(numThreads);
            List<Future<?>> futures = new ArrayList<>();
            
            for (int t = 0; t < numThreads; t++) {
                final int start = t * tasksPerThread;
                final int end = (t == numThreads - 1) ? dimSize : (t + 1) * tasksPerThread;
                
                if (start >= dimSize) break;
                
                futures.add(pool.submit(() -> {
                    for (int i = start; i < end; i++) {
                        recursivePermuteMultiBlock(
                            inDB, outDB, inDims, inStrides, permutedStrides,
                            1,
                            i * inStrides[0],
                            i * permutedStrides[0]
                        );
                    }
                }));
            }
            
            for (Future<?> f : futures) {
                try {
                    f.get();
                } catch (Exception e) {
                    throw new RuntimeException("Parallel permute failed", e);
                }
            }
            pool.shutdown();
        }
    }
}