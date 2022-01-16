package org.apache.sysds.runtime.iogen.template;

import org.apache.sysds.runtime.iogen.CustomProperties;
import org.apache.sysds.runtime.iogen.codegen.RowColIdentify;

public class TemplateCodeGenMatrix extends TemplateCodeGenBase {

	private String type;

	public TemplateCodeGenMatrix(CustomProperties properties, String className) {
		super(properties, className);

		// 1. set java code template
		// 2. set cpp code template
		javaTemplate =//"package org.apache.sysds.runtime.iogen; \n"+
			"import org.apache.commons.lang.mutable.MutableInt;\n" +
			"import org.apache.sysds.runtime.io.IOUtilFunctions;\n" +
			"import org.apache.sysds.runtime.iogen.ColumnIdentifyProperties;\n" +
			"import org.apache.sysds.runtime.iogen.CustomProperties;\n" +
			"import org.apache.sysds.runtime.matrix.data.MatrixBlock;\n" +
			"import org.apache.sysds.runtime.matrix.data.Pair;\n" +
			"import org.apache.sysds.runtime.util.UtilFunctions;\n" +
			"import org.apache.sysds.runtime.iogen.template.MatrixGenerateReader; \n"+
			"import java.io.BufferedReader;\n" +
			"import java.io.IOException;\n" +
			"import java.io.InputStream;\n" +
			"import java.io.InputStreamReader;\n" +
			"import java.util.Arrays;\n" +
			"import java.util.Comparator;\n" +

			"public class "+className+" extends MatrixGenerateReader {\n"+

			"	public "+className+"(CustomProperties _props) {\n"+
			"		super(_props);\n"+
			"	}\n"+

			"	@Override protected long readMatrixFromInputStream(InputStream is, String srcInfo, MatrixBlock dest,\n"+
			"		MutableInt rowPos, long rlen, long clen, int blen) throws IOException {\n"+
			code+
			"}}\n";

		type = properties.getRowIndex().toString() + properties.getColIndex().toString();
		switch(type){
			case "IDENTIFYIDENTIFY":
				codeGenClass = new RowColIdentify(properties, className);
				break;
			default:
				throw new RuntimeException("The properties of row and column index are not defined!!");
		}
	}

	@Override
	public String generateCodeJava() {
		return javaTemplate.replace(code, codeGenClass.generateCodeJava());
	}

	@Override public String generateCodeCPP() {
		return codeGenClass.generateCodeCPP();
	}
}
