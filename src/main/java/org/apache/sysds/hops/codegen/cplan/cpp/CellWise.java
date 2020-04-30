package org.apache.sysds.hops.codegen.cplan.cpp;

import org.apache.sysds.hops.codegen.cplan.CNodeBinary;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary;
import org.apache.sysds.hops.codegen.cplan.CodeTemplate;

public class CellWise implements CodeTemplate {
    public static final String TEMPLATE =
            "%TMP%\n"
                    + "template<typename T>\n"
                    + "	__device__ T genexec(T a, T** b, T* scalars, int m, int n, int grix, int rix, int cix) {\n"
                    + "%BODY_dense%"
                    + "    return %OUT%;\n"
                    + "	}\n"
                    + "template<typename T>\n"
                    + "	__global__\n"
                    + " 	void %TMP% (T *a, T** b, T* c, T* scalars, int m, int n, int grix, int rix, int cix) {\n"
                    + "		int tid = threadIdx.y * blockDim.x + threadIdx.x;\n" // ToDo: correct indexing!
                    + "		if(threadIdx.x < m && threadIdx.y < n) {\n"
//                    + "  		if(tid == 1)\n"
//					+ " 		    printf(\"MxN=%dx%d, a[0]=%f, scalars[0]=%f, b[0]=%f\\n\",m, n, a[0], scalars[0], b[0]);\n"
//                    + " 		    printf(\"MxN=%dx%d, a[0][%d]=%f\\n\",m, n, tid, a[0][tid]);\n"
                    + "			c[tid] = genexec(a[tid], b, scalars, m, n, grix, threadIdx.x, threadIdx.y);\n"
//                    + "         if(tid == 1)\n"
//					+ " 			printf(\"c[%d]=%f\\n\", tid, c[tid]);\n"
                    + "		}\n"
                    + "	}\n";

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
