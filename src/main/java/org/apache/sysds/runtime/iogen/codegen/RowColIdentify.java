package org.apache.sysds.runtime.iogen.codegen;

import org.apache.sysds.runtime.iogen.CustomProperties;

public class RowColIdentify extends CodeGenBase {

	public RowColIdentify(CustomProperties properties, String className) {
		super(properties, className);
	}

	@Override public String generateCodeJava() {
		String code = 			"String str; \n"+
			"int row = rowPos.intValue(); \n"+
			"double cellValue; \n"+
			"ColumnIdentifyProperties[] colsPro = _props.getColumnIdentifyProperties(); \n"+
			"int col = colsPro.length; \n"+
			"int start; \n"+
			"long lnnz = 0; \n"+

			"BufferedReader br = new BufferedReader(new InputStreamReader(is)); \n"+
			"Arrays.sort(colsPro, Comparator.comparing(ColumnIdentifyProperties::getIndexPosition)); \n"+

			"int lastIndex = 0; \n"+
			"for(ColumnIdentifyProperties cip : _props.getColumnIdentifyProperties()) { \n"+
			"	cip.setIndexPosition(cip.getIndexPosition() - lastIndex); \n"+
			"	lastIndex += cip.getIndexPosition(); \n"+
			"}\n"+
//
//			"// Read the data\n"+
//			"try {\n"+
//			"	while((str = br.readLine()) != null) //foreach line\n"+
//			"	{\n"+
//			"start = 0; \n"+
//			"for(int c = 0; + c < col; c++) {\n"+
//			"	Pair<String, Integer> pair = _props.getValue(str, start, colsPro[c].getIndexPositionDelimiter(),\n"+
//			"colsPro[c].getIndexPosition(), colsPro[c].getValueEndWithString()); \n"+
//
//			"	if(pair!=null) {\n"+
//			"cellValue = UtilFunctions.getDouble(pair.getKey()); \n"+
//			"if(cellValue != 0) {\n"+
//			"	dest.appendValue(row, col, cellValue); \n"+
//			"	lnnz++; \n"+
//			"	start += pair.getValue(); \n"+
//			"}\n"+
//			"	}\n"+
//			"	else\n"+
//			"break; \n"+
//			"}\n"+
//			"row++; \n"+
//			"	}\n"+
//			"}\n"+
//			"finally {\n"+
//			"	IOUtilFunctions.closeSilently(br); \n"+
//			"}\n"+
//			"rowPos.setValue(row); \n"+
//			"return lnnz; ";
			"return 0; ";

		return code;
	}

	@Override public String generateCodeCPP() {
		return null;
	}
}
