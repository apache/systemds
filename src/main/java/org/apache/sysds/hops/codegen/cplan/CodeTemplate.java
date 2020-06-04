package org.apache.sysds.hops.codegen.cplan;

import org.apache.sysds.runtime.codegen.SpoofCellwise;

public interface CodeTemplate {

    String getTemplate();

    String getTemplate(CNodeUnary.UnaryType type, boolean sparse);

    String getTemplate(CNodeBinary.BinType type, boolean sparseLhs, boolean sparseRhs, boolean scalarVector,
                              boolean scalarInput);

    String getTemplate(SpoofCellwise.CellType ct);
}
