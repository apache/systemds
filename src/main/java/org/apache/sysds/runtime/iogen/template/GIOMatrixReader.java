package org.apache.sysds.runtime.iogen.template;

import org.apache.commons.lang.mutable.MutableInt;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.iogen.CustomProperties;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;

public class GIOMatrixReader extends MatrixGenerateReader {

	public GIOMatrixReader(CustomProperties _props) {
		super(_props);
	}

	@Override protected long readMatrixFromInputStream(InputStream is, String srcInfo, MatrixBlock dest,
		MutableInt rowPos, long rlen, long clen, int blen) throws IOException {

		String str;
		int row = rowPos.intValue();
		double cellValue;
		long lnnz = 0;

		ArrayList<String>[] colKeyPattern = _props.getColKeyPattern();
		HashSet<String>[] endWithValueString = _props.getEndWithValueString();
		int col = endWithValueString.length;
		int index;

		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		try {
			while((str = br.readLine()) != null) //foreach line
			{
//				for(int c = 0; c < col; c++) {
//					cellValue = getCellValue(str, colKeyPattern[c], endWithValueString[c]);
//					if(cellValue != 0) {
//						dest.appendValue(row, col, cellValue);
//						lnnz++;
//					}
//				}


				row++;
			}
		}
		finally {
			IOUtilFunctions.closeSilently(br);
		}
		//
		//
		//		//-------------------------------------------------------
		//		Arrays.sort(colsPro, Comparator.comparing(ColumnIdentifyProperties::getIndexPosition));
		//
		//		int lastIndex = 0;
		//		for(ColumnIdentifyProperties cip : _props.getColumnIdentifyProperties()) {
		//			cip.setIndexPosition(cip.getIndexPosition() - lastIndex);
		//			lastIndex += cip.getIndexPosition();
		//		}
		//
		//		// Read the data
		//		try {
		//			while((str = br.readLine()) != null) //foreach line
		//			{
		//				start = 0;
		//				for(int c = 0; c < col; c++) {
		//					Pair<String, Integer> pair = _props.getValue(str, start, colsPro[c].getIndexPositionDelimiter(),
		//						colsPro[c].getIndexPosition(), colsPro[c].getValueEndWithString());
		//
		//					if(pair!=null) {
		//						cellValue = UtilFunctions.getDouble(pair.getKey());
		//						if(cellValue != 0) {
		//							dest.appendValue(row, col, cellValue);
		//							lnnz++;
		//							start += pair.getValue();
		//						}
		//					}
		//					else
		//						break;
		//				}
		//				row++;
		//			}
		//		}
		//		finally {
		//			IOUtilFunctions.closeSilently(br);
		//		}

		rowPos.setValue(row);
		return lnnz;
	}
}
