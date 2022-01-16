package org.apache.sysds.runtime.iogen.codegen;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.iogen.CustomProperties;

import java.util.ArrayList;

public class CodeGen {

	protected CustomProperties properties;
	protected String className;

	public CodeGen(CustomProperties properties, String className) {
		this.properties = properties;
		this.className = className;
	}

	public String generateCodeJava(){
		StringBuilder src = new StringBuilder();
		src.append("String str; \n");
		src.append("int row = rowPos.intValue(); \n");
		src.append("long lnnz = 0; \n");
		src.append("int index, endPos, strLen; \n");
		src.append("HashSet<String>[] endWithValueString = _props.getEndWithValueString(); \n");
		src.append("BufferedReader br = new BufferedReader(new InputStreamReader(is)); \n");
		src.append("try { \n");
		src.append("while((str = br.readLine()) != null){ \n");
		src.append("strLen = str.length(); \n");

		ArrayList<String>[] colKeyPattern = properties.getColKeyPattern();
		CodeGenTrie trie= new CodeGenTrie();
		for(int c=0; c< colKeyPattern.length; c++){
			trie.insert(c, Types.ValueType.FP64, colKeyPattern[c]);
		}
		src.append(trie.getJavaCode());

		src.append("row++; \n");
		src.append("} \n");
		src.append("} \n");
		src.append("finally { \n");
		src.append("IOUtilFunctions.closeSilently(br); \n");
		src.append("}");
		src.append("rowPos.setValue(row); \n");
		src.append("return lnnz; \n");

		return src.toString();
	}

	public String generateCodeCPP(){
		return null;
	}
}
