/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.iogen.codegen;

import org.apache.sysds.runtime.iogen.CustomProperties;
import org.apache.sysds.runtime.iogen.template.TemplateCodeGenBase;

public class MatrixCodeGen extends TemplateCodeGenBase {

	public MatrixCodeGen(CustomProperties properties, String className) {
		super(properties, className);

		// 1. set java code template
		javaTemplate = "import org.apache.commons.lang.mutable.MutableInt;\n" +
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
		// 2. set cpp code template
	}

	@Override
	public String generateCodeJava() {
		StringBuilder src = new StringBuilder();
		src.append("String str; \n");
		src.append("int row = rowPos.intValue(); \n");
		src.append("long lnnz = 0; \n");
		src.append("int index, endPos, strLen; \n");
		src.append("HashSet<String>[] endWithValueString = _props.endWithValueStrings(); \n");
		src.append("BufferedReader br = new BufferedReader(new InputStreamReader(is)); \n");
		if(properties.getRowIndex() == CustomProperties.IndexProperties.PREFIX)
			src.append("HashSet<String> endWithValueStringRow = _props.endWithValueStringsRow(); \n");
//		src.append("try { \n");
//		src.append("while((str = br.readLine()) != null){ \n");
//		src.append("strLen = str.length(); \n");

		CodeGenTrie trie= new CodeGenTrie(properties, "dest.appendValue");
		src.append(trie.getJavaCode());

//		src.append("} \n");
//		src.append("} \n");
//		src.append("finally { \n");
//		src.append("IOUtilFunctions.closeSilently(br); \n");
//		src.append("}");
		src.append("rowPos.setValue(row); \n");
		src.append("return lnnz; \n");

		return javaTemplate.replace(code, src.toString());
	}

	@Override
	public String generateCodeCPP() {
		return null;
	}
}
