package org.apache.sysds.hops.codegen.cplan.java;

import org.apache.sysds.hops.codegen.cplan.CNodeBinary;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary;
import org.apache.sysds.hops.codegen.cplan.CodeTemplate;

public class CellWise implements CodeTemplate {
    public static final String TEMPLATE =
            "package codegen;\n"
                    + "import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;\n"
                    + "import org.apache.sysds.runtime.codegen.SpoofCellwise;\n"
                    + "import org.apache.sysds.runtime.codegen.SpoofCellwise.AggOp;\n"
                    + "import org.apache.sysds.runtime.codegen.SpoofCellwise.CellType;\n"
                    + "import org.apache.sysds.runtime.codegen.SpoofOperator.SideInput;\n"
                    + "import org.apache.commons.math3.util.FastMath;\n"
                    + "\n"
                    + "public final class %TMP% extends SpoofCellwise {\n"
                    + "  public %TMP%() {\n"
                    + "    super(CellType.%TYPE%, %SPARSE_SAFE%, %SEQ%, %AGG_OP%);\n"
                    + "  }\n"
                    + "  protected double genexec(double a, SideInput[] b, double[] scalars, int m, int n, long grix, int rix, int cix) { \n"
                    + "%BODY_dense%"
                    + "    return %OUT%;\n"
                    + "  }\n"
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
