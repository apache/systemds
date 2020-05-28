package org.apache.sysds.hops.codegen.cplan.cpp;

import org.apache.sysds.hops.codegen.cplan.CNodeBinary;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary;
import org.apache.sysds.hops.codegen.cplan.CodeTemplate;

public class CellWise implements CodeTemplate {
    public static final String TEMPLATE =
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
                    + "     for(int i = first_idx; i < last_idx; i++) {\n"
                    + "         int row = i / m;\n"
                    + "         int col = i % n;\n"
                    + "         c[i] = genexec(a[i], b, scalars, m, n, grix, row, col);\n"
                    + "     }\n"
                    + "}\n";

    @Override
    public String getTemplate() {
        return TEMPLATE;
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
