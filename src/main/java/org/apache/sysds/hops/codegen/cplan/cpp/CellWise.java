package org.apache.sysds.hops.codegen.cplan.cpp;

import org.apache.sysds.hops.codegen.cplan.CNodeBinary;
import org.apache.sysds.hops.codegen.cplan.CNodeTernary;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary;
import org.apache.sysds.hops.codegen.cplan.CodeTemplate;
import org.apache.sysds.runtime.codegen.SpoofCellwise;
import org.apache.sysds.runtime.io.IOUtilFunctions;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

// ToDo: clean code template and load from file
public class CellWise implements CodeTemplate {

    private static final String TEMPLATE_PATH = "/spoof-templates/cellwise.cu";

    @Override
    public String getTemplate() {
        throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
    }

    @Override
    public String getTemplate(SpoofCellwise.CellType ct) {

        try {
            // Try loading from jar file first or from source tree otherwise
            // The latter might be the case when running SystemDS from IDE
            if(new File(TEMPLATE_PATH).isFile())
                return IOUtilFunctions.toString(getClass().getResourceAsStream(TEMPLATE_PATH));
            else
                return IOUtilFunctions.toString(new FileInputStream("src/main/cuda" + TEMPLATE_PATH));
        }
        catch(IOException e) {
            System.out.println(e.getMessage());
            return null;
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
