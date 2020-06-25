package org.apache.sysds.hops.codegen.cplan.cpp;

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary;
import org.apache.sysds.hops.codegen.cplan.CNodeTernary;
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
            "// SparseSafe: %SPARSE_SAFE%\n" +
            "// SEQ: %SEQ%\n" +
            "#include \"spoof_utils.cuh\"\n" +
            "\n"
                + "template<typename T>\n"
                + "__device__ T genexec(T a, T** b, T* scalars, int m, int n, int grix, int rix, int cix) {\n"
                + "%BODY_dense%"
                + "    return %OUT%;\n"
                + "}\n"
                + "\n"
                + "template<typename T, int VT>\n"
                + "__global__\n"
                + "void %TMP% (T *a, T** b, T* c, T* scalars, int m, int n, int grix) {\n"
                + "     int tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
                + "     int first_idx = tid * VT;\n"
                + "     int last_idx = min(first_idx + VT, m * n);\n"
                + "     #pragma unroll\n"
                + "     for(int i = first_idx; i < last_idx; i++) {\n"
                + "         int row = i / n;\n"
                + "         int col = i % n;\n"
                + "         c[i] = genexec(a[i], b, scalars, m, n, grix + row, row, col);\n"
                + "     }\n"
                + "}\n";

    public static final String TEMPLATE_FULL_AGG =
            "%TMP%\n" +
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
            "   int m, n, grix_;\n" +
            "   SpoofCellwiseOp(T** b, T* scalars, int m, int n, int grix) : \n" +
            "       b(b), scalars(scalars), m(m), n(n), grix_(grix) {}\n" +
            "   __device__  __forceinline__ T operator()(T a, int idx) const {\n" +
            "   int rix = idx / n;\n" +
            "   int cix = idx % n;\n" +
            "   int grix = grix_ + rix;\n" +
            "%BODY_dense%" +
//                    "__syncthreads();\n" +
//                    "printf(\"idx=%d, grix=%d, rix=%d, cix=%d, m=%d, n=%d, out=%f\\n\", idx, grix, rix, cix, m, n, %OUT%);\n" +
            "       return %OUT%;\n" +
            "   }\n" +
            "};" +
            "\n" +
            "template<typename T>\n" +
            "__global__ void %TMP% (T *a, T** b, T* c, T* scalars, int m, int n, int grix) {\n" +
            "   %AGG_TMPL%<T> agg_op;\n" +
            "   SpoofCellwiseOp<T> spoof_op(b, scalars, m, n, grix);\n" +
            "   reduce<T, %AGG_TMPL%<T>, SpoofCellwiseOp<T>>(a, c, m*n, %INITIAL_VALUE%, agg_op, spoof_op);\n" +
            "}\n";




    public static final String TEMPLATE_ROW_AGG = "\n";

    public static final String TEMPLATE_COL_AGG =
        "%TMP%\n" +
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
            "   int m, n, grix_;\n" +
            "   SpoofCellwiseOp(T** b, T* scalars, int m, int n, int grix) : \n" +
            "       b(b), scalars(scalars), m(m), n(n), grix_(grix) {}\n" +
            "   __device__  __forceinline__ T operator()(T a, int idx) const {\n" +
            "   int rix = idx / n;\n" +
            "   int cix = idx % n;\n" +
            "   int grix = grix_ + rix;\n" +
            "%BODY_dense%" +
            //                    "__syncthreads();\n" +
            //                    "printf(\"idx=%d, grix=%d, rix=%d, cix=%d, m=%d, n=%d, out=%f\\n\", idx, grix, rix, cix, m, n, %OUT%);\n" +
            "       return %OUT%;\n" +
            "   }\n" +
            "};" +
            "\n" +
            "template<typename T>\n" +
            "__global__ void %TMP% (T *a, T** b, T* c, T* scalars, int m, int n, int grix) {\n" +
            "   %AGG_TMPL%<T> agg_op;\n" +
            "   SpoofCellwiseOp<T> spoof_op(b, scalars, m, n, grix);\n" +
            "   reduce_col<T, %AGG_TMPL%<T>, SpoofCellwiseOp<T>>(a, c, m, n, %INITIAL_VALUE%, agg_op, spoof_op);\n" +
            "}\n";

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

    @Override
    public String getTemplate(CNodeTernary.TernaryType type, boolean sparse) {
        throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
    }
}
