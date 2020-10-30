package org.apache.sysds.hops.codegen.cplan.cpp;

import org.apache.sysds.hops.codegen.cplan.CNodeBinary;
import org.apache.sysds.hops.codegen.cplan.CNodeTernary;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary;
import org.apache.sysds.hops.codegen.cplan.CodeTemplate;
import org.apache.sysds.runtime.codegen.SpoofCellwise;

import static org.apache.sysds.runtime.matrix.data.LibMatrixNative.isSinglePrecision;

public class Ternary implements CodeTemplate {

    @Override
    public String getTemplate(CNodeTernary.TernaryType type, boolean sparse) {
        if(isSinglePrecision()) {
            switch (type) {
                case PLUS_MULT:
                    return "    T %TMP% = %IN1% + %IN2% * %IN3%;\n";

                case MINUS_MULT:
                    return "    T %TMP% = %IN1% - %IN2% * %IN3%;\n";

                case BIASADD:
                    return "    T %TMP% = %IN1% + getValue(%IN2%, cix/%IN3%);\n";

                case BIASMULT:
                    return "    T %TMP% = %IN1% * getValue(%IN2%, cix/%IN3%);\n";

                case REPLACE:
                    return "    T %TMP% = (%IN1% == %IN2% || (isnan(%IN1%) "
                            + "&& isnan(%IN2%))) ? %IN3% : %IN1%;\n";

                case REPLACE_NAN:
                    return "    T %TMP% = isnan(%IN1%) ? %IN3% : %IN1%;\n";

                case IFELSE:
                    return "    T %TMP% = (%IN1% != 0) ? %IN2% : %IN3%;\n";

                case LOOKUP_RC1:
                    return sparse ?
                            "    T %TMP% = getValue(%IN1v%, %IN1i%, ai, alen, %IN3%-1);\n" :
                            "    T %TMP% = getValue(%IN1%, %IN2%, rix, %IN3%-1);\n";

                case LOOKUP_RVECT1:
                    return "    T[] %TMP% = getVector(%IN1%, %IN2%, rix, %IN3%-1);\n";

                default:
                    throw new RuntimeException("Invalid ternary type: " + this.toString());
            }
        }
        else {
            switch (type) {
                case PLUS_MULT:
                    return "    T %TMP% = %IN1% + %IN2% * %IN3%;\n";

                case MINUS_MULT:
                    return "    T %TMP% = %IN1% - %IN2% * %IN3%;\n";

                case BIASADD:
                    return "    T %TMP% = %IN1% + getValue(%IN2%, cix/%IN3%);\n";

                case BIASMULT:
                    return "    T %TMP% = %IN1% * getValue(%IN2%, cix/%IN3%);\n";

                case REPLACE:
                    return "    T %TMP% = (%IN1% == %IN2% || (isnan(%IN1%) "
                            + "&& isnan(%IN2%))) ? %IN3% : %IN1%;\n";

                case REPLACE_NAN:
                    return "    T %TMP% = isnan(%IN1%) ? %IN3% : %IN1%;\n";

                case IFELSE:
                    return "    T %TMP% = (%IN1% != 0) ? %IN2% : %IN3%;\n";

                case LOOKUP_RC1:
                    return sparse ?
                            "    T %TMP% = getValue(%IN1v%, %IN1i%, ai, alen, %IN3%-1);\n" :
                            "    T %TMP% = getValue(%IN1%, %IN2%, rix, %IN3%-1);\n";

                case LOOKUP_RVECT1:
                    return "    T[] %TMP% = getVector(%IN1%, %IN2%, rix, %IN3%-1);\n";

                default:
                    throw new RuntimeException("Invalid ternary type: "+this.toString());
            }

        }
    }

    @Override
    public String getTemplate(CNodeBinary.BinType type, boolean sparseLhs, boolean sparseRhs, boolean scalarVector,
                              boolean scalarInput) {
        throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
    }

    @Override
    public String getTemplate() {
        throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
    }

    @Override
    public String getTemplate(SpoofCellwise.CellType ct) {
        throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
    }

    @Override
    public String getTemplate(CNodeUnary.UnaryType type, boolean sparse) {
        throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
    }

}
