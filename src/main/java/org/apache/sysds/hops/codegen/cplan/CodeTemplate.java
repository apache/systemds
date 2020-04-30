package org.apache.sysds.hops.codegen.cplan;

public interface CodeTemplate {

    String getTemplate();

    String getTemplate(CNodeUnary.UnaryType type, boolean sparse);

    String getTemplate(CNodeBinary.BinType type, boolean sparseLhs, boolean sparseRhs, boolean scalarVector,
                              boolean scalarInput);

}
