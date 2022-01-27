package org.apache.sysds.runtime.iogen.codegen;

import org.apache.commons.lang.mutable.MutableInt;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.iogen.CustomProperties;
import org.apache.sysds.runtime.iogen.template.MatrixGenerateReader;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashSet;

public class myTest extends MatrixGenerateReader {
	public myTest(CustomProperties _props) {
		super(_props);
	}

	@Override protected long readMatrixFromInputStream(InputStream is, String srcInfo, MatrixBlock dest,
		MutableInt rowPos, long rlen, long clen, int blen) throws IOException {

		String str;
		int row = rowPos.intValue();
		long lnnz = 0;
		int index, endPos, strLen;
		HashSet<String>[] endWithValueString = _props.endWithValueStrings();
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		String strChunk, remainedStr = null;
		int chunkSize = 2048;
		int recordIndex = 0;
		try {
			do{
				strChunk = getStringChunkOfBufferReader(br, remainedStr, chunkSize);
				System.out.println(strChunk);
				if(strChunk == null || strChunk.length() == 0) break;
				do {
					recordIndex = strChunk.indexOf("#index ", recordIndex);
					if(recordIndex == -1) break;
					//recordIndex +=7;
					int recordBeginPos = recordIndex;
					recordIndex = strChunk.indexOf("#index ", recordBeginPos + 7);if(recordIndex == -1) break;
					str = strChunk.substring(recordBeginPos, recordIndex);
					strLen = str.length();
					index = str.indexOf(" ");
					if(index != -1) {
						int curPos_75540091 = index + 1;
						endPos = getEndPos(str, strLen, curPos_75540091, endWithValueString[1]);
						String cellStr1 = str.substring(curPos_75540091,endPos);
						if ( cellStr1.length() > 0 ){
							Double cellValue1;
							try{ cellValue1= Double.parseDouble(cellStr1); } catch(Exception e){cellValue1 = 0d;}
							if(cellValue1 != 0) {
								dest.appendValue(row, 1, cellValue1);
								lnnz++;
							}
						}
					}
					index = str.indexOf("#index ");
					if(index != -1) {
						int curPos_50855160 = index + 7;
						endPos = getEndPos(str, strLen, curPos_50855160, endWithValueString[0]);
						String cellStr0 = str.substring(curPos_50855160,endPos);
						if ( cellStr0.length() > 0 ){
							Double cellValue0;
							try{ cellValue0= Double.parseDouble(cellStr0); } catch(Exception e){cellValue0 = 0d;}
							if(cellValue0 != 0) {
								dest.appendValue(row, 0, cellValue0);
								lnnz++;
							}
						}
					}
					index = str.indexOf("#index 1");
					if(index != -1) {
						int curPos_36575074 = index + 8;
						index = str.indexOf(",", curPos_36575074);
						if(index != -1) {
							int curPos_13302308 = index + 1;
							endPos = getEndPos(str, strLen, curPos_13302308, endWithValueString[2]);
							String cellStr2 = str.substring(curPos_13302308,endPos);
							if ( cellStr2.length() > 0 ){
								Double cellValue2;
								try{ cellValue2= Double.parseDouble(cellStr2); } catch(Exception e){cellValue2 = 0d;}
								if(cellValue2 != 0) {
									dest.appendValue(row, 2, cellValue2);
									lnnz++;
								}
							}
						}
					}
					row++;
				}while(true);
				remainedStr = strChunk.substring(recordIndex);
			}while(true);
		}
		finally {
			IOUtilFunctions.closeSilently(br);
		}
		rowPos.setValue(row);
		return lnnz;
	}
}
