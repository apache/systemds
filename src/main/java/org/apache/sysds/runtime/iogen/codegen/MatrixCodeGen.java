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

import org.apache.sysds.runtime.iogen.ColIndexStructure;
import org.apache.sysds.runtime.iogen.CustomProperties;
import org.apache.sysds.runtime.iogen.FormatIdentifyer;
import org.apache.sysds.runtime.iogen.RowIndexStructure;
import org.apache.sysds.runtime.iogen.template.TemplateCodeGenBase;

public class MatrixCodeGen extends TemplateCodeGenBase {

	public MatrixCodeGen(CustomProperties properties, String className) {
		super(properties, className);

		// 1. set java code template
		String javaBaseClass = !properties.isParallel() ? "MatrixGenerateReader" : "MatrixGenerateReaderParallel";
		javaTemplate = "import org.apache.hadoop.io.LongWritable;\n" +
			"import org.apache.hadoop.io.Text;\n" +
			"import org.apache.hadoop.mapred.RecordReader;\n" +
			"import org.apache.sysds.runtime.iogen.CustomProperties;\n" +
			"import org.apache.sysds.runtime.iogen.template."+javaBaseClass+";\n" +
			"import org.apache.sysds.runtime.matrix.data.MatrixBlock;\n" +
			"import java.io.IOException;\n" +
			"import java.util.HashSet;\n" +
			"import org.apache.sysds.runtime.io.IOUtilFunctions; \n" +
			"import org.apache.sysds.runtime.util.UtilFunctions; \n" +
			"import org.apache.commons.lang.mutable.MutableInt; \n" +
			"import org.apache.sysds.runtime.iogen.template.TemplateUtil; \n" +
			"public class "+className+" extends "+javaBaseClass+" {\n" +
			"public "+className+"(CustomProperties _props) {\n" +
			"super(_props);} \n" +
			"@Override \n" +
			"protected long readMatrixFromHDFS(RecordReader<LongWritable, Text> reader, LongWritable key, Text value, " +
			"MatrixBlock dest, MutableInt rowPos, TemplateUtil.SplitInfo splitInfo) throws IOException {\n"+
			code+
			"}} \n";

		// 2. set cpp code template
	}

	@Override
	public String generateCodeJava(FormatIdentifyer formatIdentifyer) {

		StringBuilder src = new StringBuilder();
		CodeGenTrie trie = new CodeGenTrie(properties, "dest.appendValue", true, formatIdentifyer);
		src.append("String str=\"\"; \n");
		src.append("String remainStr = \"\"; \n");
		src.append("int col = -1; \n");
		src.append("String[] parts; \n");
		src.append("int row = rowPos.intValue(); \n");
		src.append("long lnnz = 0; \n");
		src.append("int index, endPos, strLen; \n");

		boolean flag1 = false;
		boolean flag2 = false;

		if( properties.getRowIndexStructure().getProperties() == RowIndexStructure.IndexProperties.RowWiseExist ||
			properties.getRowIndexStructure().getProperties() == RowIndexStructure.IndexProperties.CellWiseExist) {
			src.append("HashSet<String> endWithValueStringRow = _props.getRowIndexStructure().endWithValueStrings(); \n");
			flag1 = true;
		}

		if(properties.getColIndexStructure().getProperties() == ColIndexStructure.IndexProperties.CellWiseExist) {
			src.append("HashSet<String> endWithValueStringCol = _props.getColIndexStructure().endWithValueStrings(); \n");
			flag2 = true;
		}

		if(flag1 && flag2)
			src.append("HashSet<String> endWithValueStringVal = _props.endWithValueStrings()[0]; \n");
		else
			src.append("HashSet<String>[] endWithValueString = _props.endWithValueStrings(); \n");

		src.append("try { \n");
		if(properties.getRowIndexStructure().getProperties() == RowIndexStructure.IndexProperties.SeqScatter){
			src.append("int ri = -1; \n");
			src.append("int beginPosStr, endPosStr; \n");
			src.append("StringBuilder sb = new StringBuilder(); \n");
			src.append("long beginIndex = splitInfo.getRecordIndexBegin(0); \n");
			src.append("long endIndex = splitInfo.getRecordIndexEnd(0); \n");
			src.append("boolean flag = true; \n");
			src.append("while(flag || sb.length() > 0) { \n");
			src.append("flag = reader.next(key, value); \n");
			src.append("if(flag) { \n");
			src.append("ri++; \n");
			src.append("String valStr = value.toString(); \n");
			src.append("beginPosStr = ri == beginIndex ? splitInfo.getRecordPositionBegin(row) : 0; \n");
			src.append("endPosStr = ri == endIndex ? splitInfo.getRecordPositionEnd(row): valStr.length(); \n");
			src.append("if(ri >= beginIndex && ri <= endIndex){ \n");
			src.append("sb.append(valStr.substring(beginPosStr, endPosStr)); \n");
			src.append("remainStr = valStr.substring(endPosStr); \n");
			src.append("continue; \n");
			src.append("} \n");
			src.append("else { \n");
			src.append("str = sb.toString(); \n");
			src.append("sb = new StringBuilder(); \n");
			src.append("sb.append(remainStr).append(valStr); \n");
			src.append("beginIndex = splitInfo.getRecordIndexBegin(row+1); \n");
			src.append("endIndex = splitInfo.getRecordIndexEnd(row+1); \n");
			src.append("} \n");
			src.append("} \n");
			src.append("else {\n");
			src.append("str = sb.toString(); \n");
			src.append("sb = new StringBuilder();\n");
			src.append("}");
		}
		else {
			src.append("while(reader.next(key, value)) { \n");
			src.append("str = value.toString(); \n");
		}
		src.append("strLen = str.length(); \n");
		src.append(trie.getJavaCode());
		src.append("} \n");
		src.append("} \n");
		src.append("catch(Exception ex){ \n");
		src.append("} \n");
		src.append("rowPos.setValue(row); \n");
		src.append("return lnnz; \n");
		return javaTemplate.replace(code, src.toString());
	}

	@Override
	public String generateCodeCPP() {
		return null;
	}
}
