package org.apache.sysds.hops.codegen.cplan.cpp;

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary;
import org.apache.sysds.hops.codegen.cplan.CodeTemplate;
import org.apache.sysds.runtime.codegen.SpoofCellwise;

import static org.apache.sysds.runtime.codegen.SpoofCellwise.CellType.NO_AGG;

// ToDo: clean code template and load from file
public class CellWise implements CodeTemplate {
    public static final String TEMPLATE_NO_AGG =
            "%TMP%\n" +
            "// CellType: %TYPE%\n" +
            "// AggOp: %AGG_OP%\n" +
            "// SparseSafe: $SPARSE_SAFE%\n" +
            "// SEQ: %SEQ%\n"
                + "template<typename T>\n"
                + "__device__ T getValue(T* data, int rowIndex) {\n"
                + "    return data[rowIndex];\n"
                + "}\n"
                + "\n"
                + "template<typename T>\n"
                + "__device__ T getValue(T* data, int n, int rowIndex, int colIndex) {\n"
                + "     return data[rowIndex * n + colIndex];\n"
                + "}\n"
                    + "template<typename T>\n"
                    + "__device__ T modulus(T a, T b) {\n" +
                    "   return a - a / b * b;\n"
                    + "//     return *reinterpret_cast<int*>(&a) & *reinterpret_cast<int*>(&b);\n"
                    + "}\n"
                    + "template<typename T>\n"
                    + "__device__ T intDiv(T a, T b) {\n" +
                    "   T tmp = a / b;\n" +
                    "   return *reinterpret_cast<int*>(&tmp);"
                    + "//     return *reinterpret_cast<int*>(&a) & *reinterpret_cast<int*>(&b);\n"
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
            "// CellType: %TYPE%\n" +
            "// AggOp: %AGG_OP%\n" +
            "// SparseSafe: %SPARSE_SAFE%\n" +
            "// SEQ: %SEQ%\n" +
            "#include \"utils.cuh\"\n" +
            "#include \"agg_ops.cuh\"\n" +
            "#include \"spoof_utils.cuh\"\n" +
            "#include \"reduction.cuh\"\n" +
            "\n" +
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
