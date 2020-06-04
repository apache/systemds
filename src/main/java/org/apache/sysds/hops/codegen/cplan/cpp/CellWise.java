package org.apache.sysds.hops.codegen.cplan.cpp;

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary;
import org.apache.sysds.hops.codegen.cplan.CodeTemplate;
import org.apache.sysds.runtime.codegen.SpoofCellwise;

import static org.apache.sysds.runtime.codegen.SpoofCellwise.CellType.NO_AGG;

public class CellWise implements CodeTemplate {
    public static final String TEMPLATE_NO_AGG =
            "%TMP%\n"
                + "template<typename T>\n"
                + "__device__ T getValue(T* data, int rowIndex) {\n"
                + "    return data[rowIndex];\n"
                + "}\n"
                + "\n"
                + "template<typename T>\n"
                + "__device__ T getValue(T* data, int n, int rowIndex, int colIndex) {\n"
                + "     return data[rowIndex * n + colIndex];\n"
                + "}\n"
                + "\n"
                    + "template<typename T>\n"
                    + "__device__ T genexec(T a, T** b, T* scalars, int m, int n, int grix, int rix, int cix) {\n"
                    + "%BODY_dense%"
                    + "    return %OUT%;\n"
                    + "}\n"
                    + "\n"
                    + "template<typename T, int VT>\n"
                    + "__global__\n"
                    + "void %TMP% (T *a, T** b, T* c, T* scalars, int m, int n, int grix, int rix, int cix) {\n"
                    + "     int tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
                    + "     int first_idx = tid * VT;\n"
                    + "     int last_idx = min(first_idx + VT, m * n);\n"
                    + "     #pragma unroll\n"
                    + "     for(int i = first_idx; i < last_idx; i++) {\n"
                    + "         int row = i / m;\n"
                    + "         int col = i % n;\n"
                    + "         c[i] = genexec(a[i], b, scalars, m, n, grix, row, col);\n"
                    + "     }\n"
                    + "}\n";


    public static final String TEMPLATE_FULL_AGG = "%TMP%\n" +
            "#include \"utils.cuh\"\n" +
            "#include \"agg_ops.cuh\"\n" +
            "\n"
            + "template<typename T>\n"
            + "__device__ T getValue(T* data, int rowIndex) {\n"
            + "    return data[rowIndex];\n"
            + "}\n"
            + "\n"
            + "template<typename T>\n"
            + "__device__ T getValue(T* data, int n, int rowIndex, int colIndex) {\n"
            + "     return data[rowIndex * n + colIndex];\n"
            + "}\n"
            + "\n" +
            "/**\n" +
            " * Functor op for summation operation\n" +
            " */\n" +
            "template<typename T>\n" +
            "struct SpoofCellwiseOp {\n" +
            "   T**b; T* scalars; \n" +
            "   int m, n, grix, rix, cix;\n" +
            "   SpoofCellwiseOp(T** b, T* scalars, int m, int n, int grix, int rix, int cix) : \n" +
            "       b(b), scalars(scalars), m(m), n(n), grix(grix), rix(rix), cix(cix) {}\n" +
            "   __device__  __forceinline__ T operator()(T a) const {\n" +
            "       %BODY_dense%\n" +
            "       return %OUT%;\n" +
            "   }\n" +
            "};" +
            "\n" +
            "template<typename ReductionOp, typename T, typename SpoofCellwiseOpT>\n" +
            "__device__ void reduce(T *g_idata, ///< input data stored in device memory (of size n)\n" +
            "\t\tT *g_odata, ///< output/temporary array stored in device memory (of size n)\n" +
            "\t\t int n,  ///< size of the input and temporary/output arrays\n" +
            "\t\tReductionOp reduction_op, ///< Reduction operation to perform (functor object)\n" +
            "\t\tT initialValue,     ///< initial value for the reduction variable\n" +
            "       SpoofCellwiseOpT spoof_op)\n" +
            "{\n" +
            "\tauto sdata = shared_memory_proxy<T>();\n" +
            "\n" +
            "\t// perform first level of reduction,\n" +
            "\t// reading from global memory, writing to shared memory\n" +
            "\tunsigned int tid = threadIdx.x;\n" +
            "\tunsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;\n" +
            "\tunsigned int gridSize = blockDim.x * 2 * gridDim.x;\n" +
            "\n" +
            "\tT v = initialValue;\n" +
            "\n" +
            "\t// we reduce multiple elements per thread.  The number is determined by the\n" +
            "\t// number of active thread blocks (via gridDim).  More blocks will result\n" +
            "\t// in a larger gridSize and therefore fewer elements per thread\n" +
            "\twhile (i < n) {\n" +
            "\t\tv = reduction_op(v, spoof_op(g_idata[i]));\n" +
            "\t\t// ensure we don't read out of bounds\n" +
            "\t\tif (i + blockDim.x < n)\n" +
            "\t\t\tv = reduction_op(v, spoof_op(g_idata[i + blockDim.x]));\n" +
            "\t\ti += gridSize;\n" +
            "\t}\n" +
            "\n" +
            "\t// each thread puts its local sum into shared memory\n" +
            "\tsdata[tid] = v;\n" +
            "\t__syncthreads();\n" +
            "\n" +
            "\t// do reduction in shared mem\n" +
            "\tif (blockDim.x >= 1024) {\n" +
            "\t\tif (tid < 512) {\n" +
            "\t\t\tsdata[tid] = v = reduction_op(v, sdata[tid + 512]);\n" +
            "\t\t}\n" +
            "\t\t__syncthreads();\n" +
            "\t}\n" +
            "\tif (blockDim.x >= 512) {\n" +
            "\t\tif (tid < 256) {\n" +
            "\t\t\tsdata[tid] = v = reduction_op(v, sdata[tid + 256]);\n" +
            "\t\t}\n" +
            "\t\t__syncthreads();\n" +
            "\t}\n" +
            "\tif (blockDim.x >= 256) {\n" +
            "\t\tif (tid < 128) {\n" +
            "\t\t\tsdata[tid] = v = reduction_op(v, sdata[tid + 128]);\n" +
            "\t\t}\n" +
            "\t\t__syncthreads();\n" +
            "\t}\n" +
            "\tif (blockDim.x >= 128) {\n" +
            "\t\tif (tid < 64) {\n" +
            "\t\t\tsdata[tid] = v = reduction_op(v, sdata[tid + 64]);\n" +
            "\t\t}\n" +
            "\t\t__syncthreads();\n" +
            "\t}\n" +
            "\n" +
            "\tif (tid < 32) {\n" +
            "\t\t// now that we are using warp-synchronous programming (below)\n" +
            "\t\t// we need to declare our shared memory volatile so that the compiler\n" +
            "\t\t// doesn't reorder stores to it and induce incorrect behavior.\n" +
            "\t\tvolatile T *smem = sdata;\n" +
            "\t\tif (blockDim.x >= 64) {\n" +
            "\t\t\tsmem[tid] = v = reduction_op(v, smem[tid + 32]);\n" +
            "\t\t}\n" +
            "\t\tif (blockDim.x >= 32) {\n" +
            "\t\t\tsmem[tid] = v = reduction_op(v, smem[tid + 16]);\n" +
            "\t\t}\n" +
            "\t\tif (blockDim.x >= 16) {\n" +
            "\t\t\tsmem[tid] = v = reduction_op(v, smem[tid + 8]);\n" +
            "\t\t}\n" +
            "\t\tif (blockDim.x >= 8) {\n" +
            "\t\t\tsmem[tid] = v = reduction_op(v, smem[tid + 4]);\n" +
            "\t\t}\n" +
            "\t\tif (blockDim.x >= 4) {\n" +
            "\t\t\tsmem[tid] = v = reduction_op(v, smem[tid + 2]);\n" +
            "\t\t}\n" +
            "\t\tif (blockDim.x >= 2) {\n" +
            "\t\t\tsmem[tid] = v = reduction_op(v, smem[tid + 1]);\n" +
            "\t\t}\n" +
            "\t}\n" +
            "\n" +
            "\t// write result for this block to global mem\n" +
            "\tif (tid == 0)\n" +
            "\t\tg_odata[blockIdx.x] = sdata[0];\n" +
            "}\n" +
            "\n" +
            "template<typename T>\n" +
            "__global__ void %TMP% (T *a, T** b, T* c, T* scalars, int m, int n, int grix, int rix, int cix) {\n" +
            "\tSumOp<T> agg_op;\n" +
            "   SpoofCellwiseOp<T> spoof_op(b, scalars, m, n, grix, rix, cix);\n" +
            "\treduce<SumOp<T>, T, SpoofCellwiseOp<T>>(a, c, m*n, agg_op, (T) 0.0, spoof_op);\n" +
            "}\n";

    public static final String TEMPLATE_ROW_AGG = "\n";

    public static final String TEMPLATE_COL_AGG = "\n";

    @Override
    public String getTemplate() {
        throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
    }

    @Override
    public String getTemplate(SpoofCellwise.CellType ct) {
        switch(ct) {
            case NO_AGG:
                return TEMPLATE_NO_AGG;
            case FULL_AGG:
                return TEMPLATE_FULL_AGG;
            case ROW_AGG:
                return TEMPLATE_ROW_AGG;
            case COL_AGG:
                return TEMPLATE_COL_AGG;
            default:
                throw new RuntimeException("Unknown CellType: " + ct);
        }
    }

    @Override
    public String getTemplate(CNodeUnary.UnaryType type, boolean sparse) {
        throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
    }

    @Override
    public String getTemplate(CNodeBinary.BinType type, boolean sparseLhs, boolean sparseRhs, boolean scalarVector, boolean scalarInput) {
        throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
    }
}
