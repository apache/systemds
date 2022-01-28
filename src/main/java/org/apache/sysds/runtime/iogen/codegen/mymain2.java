package org.apache.sysds.runtime.iogen.codegen;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.*;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.iogen.CustomProperties;
import org.apache.sysds.runtime.iogen.template.FrameGenerateReader;
import org.apache.sysds.runtime.matrix.data.FrameBlock;

import java.io.IOException;
import java.util.HashSet;

public class mymain2 extends FrameGenerateReader {
    public mymain2(CustomProperties _props) {
        super(_props);
    }

    @Override
    protected int readFrameFromInputSplit(InputSplit split, InputFormat<LongWritable, Text> informat, JobConf job, FrameBlock dest, Types.ValueType[] schema, String[] names, long rlen, long clen, int rl, boolean first) throws IOException {
        RecordReader<LongWritable, Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
        LongWritable key = new LongWritable();
        Text value = new Text();
        int row = rl;
        long lnnz = 0;
        HashSet<String>[] endWithValueString = _props.endWithValueStrings();
        int index, endPos, strLen;
        try {
            while(reader.next(key, value)){
                String str = value.toString();
                strLen = str.length();
                endPos = getEndPos(str, strLen, 0, endWithValueString[0]);
                String cellStr0 = str.substring(0,endPos);
                if ( cellStr0.length() > 0 ){
                    Long cellValue0;
                    try{cellValue0= Long.parseLong(cellStr0); } catch(Exception e){cellValue0 = 0l;}
                    if(cellValue0 != 0) {
                        dest.set(row, 0, cellValue0);
                        lnnz++;
                    }
                }
                index = str.indexOf(",,,\"");
                if(index != -1) {
                    int curPos_60858111 = index + 4;
                    index = str.indexOf("\",", curPos_60858111);
                    if(index != -1) {
                        int curPos_2775125 = index + 2;
                        endPos = getEndPos(str, strLen, curPos_2775125, endWithValueString[6]);
                        String cellStr6 = str.substring(curPos_2775125,endPos);
                        String cellValue6 = cellStr6;
                        dest.set(row, 6, cellValue6);
                    }
                }
                index = str.indexOf(",\"");
                if(index != -1) {
                    int curPos_63087344 = index + 2;
                    endPos = getEndPos(str, strLen, curPos_63087344, endWithValueString[1]);
                    String cellStr1 = str.substring(curPos_63087344,endPos);
                    String cellValue1 = cellStr1;
                    dest.set(row, 1, cellValue1);
                }
                index = str.indexOf("\",");
                if(index != -1) {
                    int curPos_41366400 = index + 2;
                    endPos = getEndPos(str, strLen, curPos_41366400, endWithValueString[2]);
                    String cellStr2 = str.substring(curPos_41366400,endPos);
                    if ( cellStr2.length() > 0 ){
                        Integer cellValue2;
                        try{ cellValue2= Integer.parseInt(cellStr2);} catch(Exception e){cellValue2 = 0;}
                        if(cellValue2 != 0) {
                            dest.set(row, 2, cellValue2);
                            lnnz++;
                        }
                    }
                    index = str.indexOf(",", curPos_41366400);
                    if(index != -1) {
                        int curPos_78452455 = index + 1;
                        endPos = getEndPos(str, strLen, curPos_78452455, endWithValueString[3]);
                        String cellStr3 = str.substring(curPos_78452455,endPos);
                        if ( cellStr3.length() > 0 ){
                            Integer cellValue3;
                            try{ cellValue3= Integer.parseInt(cellStr3);} catch(Exception e){cellValue3 = 0;}
                            if(cellValue3 != 0) {
                                dest.set(row, 3, cellValue3);
                                lnnz++;
                            }
                        }
                    }
                }
                index = str.indexOf(",,,");
                if(index != -1) {
                    int curPos_8253849 = index + 3;
                    endPos = getEndPos(str, strLen, curPos_8253849, endWithValueString[5]);
                    String cellStr5 = str.substring(curPos_8253849,endPos);
                    String cellValue5 = cellStr5;
                    dest.set(row, 5, cellValue5);
                }
                row++;
            }}
        finally {
            IOUtilFunctions.closeSilently(reader);
        }
        return row;


    }
}
