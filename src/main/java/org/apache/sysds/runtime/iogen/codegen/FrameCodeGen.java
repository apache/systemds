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
import java.util.ArrayList;

public class FrameCodeGen extends TemplateCodeGenBase {

	public FrameCodeGen(CustomProperties properties, String className) {
		super(properties, className);

		// 1. set java code template
		javaTemplate = "import org.apache.hadoop.io.LongWritable; \n" +
		"import org.apache.hadoop.io.Text; \n" +
		"import org.apache.hadoop.mapred.InputFormat; \n" +
		"import org.apache.hadoop.mapred.InputSplit; \n" +
		"import org.apache.hadoop.mapred.JobConf; \n" +
		"import org.apache.hadoop.mapred.RecordReader; \n" +
		"import org.apache.hadoop.mapred.Reporter; \n" +
		"import org.apache.sysds.common.Types; \n" +
		"import org.apache.sysds.runtime.io.IOUtilFunctions; \n" +
		"import org.apache.sysds.runtime.iogen.CustomProperties; \n" +
		"import org.apache.sysds.runtime.matrix.data.FrameBlock; \n" +
		"import org.apache.sysds.runtime.iogen.template.FrameGenerateReader; \n" +
		"import java.io.IOException; \n" +
		"import java.util.HashSet; \n" +
		"public class "+className+" extends FrameGenerateReader{ \n" +
		"public "+className+"(CustomProperties _props) { \n" +
		"		super(_props); \n" +
		"	} \n" +

		"@Override protected int readFrameFromInputSplit(InputSplit split, InputFormat<LongWritable, Text> informat, \n" +
		"		JobConf job, FrameBlock dest, Types.ValueType[] schema, String[] names, long rlen, long clen, int rl, \n" +
		"		boolean first) throws IOException { \n" +
		code+
		"}} \n";

	}

	@Override public String generateCodeJava() {

		StringBuilder src = new StringBuilder();
		src.append("RecordReader<LongWritable, Text> reader = informat.getRecordReader(split, job, Reporter.NULL); \n");
		src.append("LongWritable key = new LongWritable(); \n");
		src.append("Text value = new Text(); \n");
		src.append("int row = rl; \n");
		src.append("long lnnz = 0; \n");
		src.append("HashSet<String>[] endWithValueString = _props.getEndWithValueString(); \n");
		src.append("int index, endPos, strLen; \n");
		src.append("try { \n");
		src.append("while(reader.next(key, value)){ \n");
		src.append("String str = value.toString(); \n");
		src.append("strLen = str.length(); \n");

		ArrayList<String>[] colKeyPattern = properties.getColKeyPattern();
		CodeGenTrie trie = new CodeGenTrie();
		for(int c = 0; c < colKeyPattern.length; c++) {
			trie.insert(c, properties.getSchema()[c], colKeyPattern[c]);
		}
		src.append(trie.getJavaCode("dest.set"));

		src.append("row++; \n");
		src.append("}} \n");
		src.append("finally { \n");
		src.append("IOUtilFunctions.closeSilently(reader); \n");
		src.append("} \n");
		src.append("return row; \n");

		return javaTemplate.replace(code, src.toString());
	}

	@Override public String generateCodeCPP() {
		return null;
	}
}
