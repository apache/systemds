package org.apache.sysds.runtime.iogen.template;

import org.apache.sysds.runtime.iogen.CustomProperties;
import org.apache.sysds.runtime.iogen.codegen.CodeGen;

public class TemplateCodeGenMatrix extends TemplateCodeGenBase {

	public TemplateCodeGenMatrix(CustomProperties properties, String className) {
		super(properties, className);

		// 1. set java code template
		// 2. set cpp code template
		javaTemplate =//"package org.apache.sysds.runtime.iogen; \n"+
					"import org.apache.commons.lang.mutable.MutableInt;\n" +
					"import org.apache.sysds.runtime.io.IOUtilFunctions;\n" +
					"import org.apache.sysds.runtime.iogen.CustomProperties;\n" +
					"import org.apache.sysds.runtime.matrix.data.MatrixBlock;\n" +
					"import org.apache.sysds.runtime.iogen.template.MatrixGenerateReader; \n" +
					"import java.io.BufferedReader;\n" +
					"import java.io.IOException;\n" +
					"import java.io.InputStream;\n" +
					"import java.io.InputStreamReader;\n" +
					"import java.util.HashSet; \n" +

					"public class "+className+" extends MatrixGenerateReader {\n"+

					"	public "+className+"(CustomProperties _props) {\n"+
					"		super(_props);\n"+
					"	}\n"+

					"	@Override protected long readMatrixFromInputStream(InputStream is, String srcInfo, MatrixBlock dest,\n"+
					"		MutableInt rowPos, long rlen, long clen, int blen) throws IOException {\n"+
					code+
					"}}\n";

		codeGen = new CodeGen(properties, className);
	}

	@Override
	public String generateCodeJava() {
		return javaTemplate.replace(code, codeGen.generateCodeJava());
	}

	@Override public String generateCodeCPP() {
		return codeGen.generateCodeCPP();
	}
}
