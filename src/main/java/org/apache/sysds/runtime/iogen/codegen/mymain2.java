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
                index = str.indexOf("\",\"\",");
                if(index != -1) {
                    int curPos_87607716 = index + 5;
                    index = str.indexOf("\",", curPos_87607716);
                    if(index != -1) {
                        int curPos_57566352 = index + 2;
                        endPos = getEndPos(str, strLen, curPos_57566352, endWithValueString[5]);
                        String cellStr5 = str.substring(curPos_57566352,endPos);
                        String cellValue5 = cellStr5;
                        dest.set(row, 5, cellValue5);
                    }
                }
                index = str.indexOf("\",19");
                if(index != -1) {
                    int curPos_8998302 = index + 4;
                    index = str.indexOf(",\"", curPos_8998302);
                    if(index != -1) {
                        int curPos_70036865 = index + 2;
                        endPos = getEndPos(str, strLen, curPos_70036865, endWithValueString[3]);
                        String cellStr3 = str.substring(curPos_70036865,endPos);
                        String cellValue3 = cellStr3;
                        dest.set(row, 3, cellValue3);
                    }
                }
                index = str.indexOf(",\"");
                if(index != -1) {
                    int curPos_33286870 = index + 2;
                    endPos = getEndPos(str, strLen, curPos_33286870, endWithValueString[1]);
                    String cellStr1 = str.substring(curPos_33286870,endPos);
                    String cellValue1 = cellStr1;
                    dest.set(row, 1, cellValue1);
                }
                index = str.indexOf("l\",");
                if(index != -1) {
                    int curPos_44381926 = index + 3;
                    endPos = getEndPos(str, strLen, curPos_44381926, endWithValueString[4]);
                    String cellStr4 = str.substring(curPos_44381926,endPos);
                    String cellValue4 = cellStr4;
                    dest.set(row, 4, cellValue4);
                }
                index = str.indexOf("\",");
                if(index != -1) {
                    int curPos_90282355 = index + 2;
                    endPos = getEndPos(str, strLen, curPos_90282355, endWithValueString[2]);
                    String cellStr2 = str.substring(curPos_90282355,endPos);
                    if ( cellStr2.length() > 0 ){
                        Integer cellValue2;
                        try{ cellValue2= Integer.parseInt(cellStr2);} catch(Exception e){cellValue2 = 0;}
                        if(cellValue2 != 0) {
                            dest.set(row, 2, cellValue2);
                            lnnz++;
                        }
                    }
                }
                index = str.indexOf(",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"");
                if(index != -1) {
                    int curPos_86635269 = index + 50;
                    index = str.indexOf(",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"", curPos_86635269);
                    if(index != -1) {
                        int curPos_4908949 = index + 50;
                        index = str.indexOf(",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"", curPos_4908949);
                        if(index != -1) {
                            int curPos_99118963 = index + 50;
                            index = str.indexOf(",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"", curPos_99118963);
                            if(index != -1) {
                                int curPos_81981300 = index + 50;
                                index = str.indexOf(",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"", curPos_81981300);
                                if(index != -1) {
                                    int curPos_7528404 = index + 50;
                                    index = str.indexOf("\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",", curPos_7528404);
                                    if(index != -1) {
                                        int curPos_55594937 = index + 50;
                                        endPos = getEndPos(str, strLen, curPos_55594937, endWithValueString[6]);
                                        String cellStr6 = str.substring(curPos_55594937,endPos);
                                        String cellValue6 = cellStr6;
                                        dest.set(row, 6, cellValue6);
                                    }
                                    index = str.indexOf(",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"", curPos_7528404);
                                    if(index != -1) {
                                        int curPos_20058273 = index + 50;
                                        index = str.indexOf("\",", curPos_20058273);
                                        if(index != -1) {
                                            int curPos_57197559 = index + 2;
                                            endPos = getEndPos(str, strLen, curPos_57197559, endWithValueString[7]);
                                            String cellStr7 = str.substring(curPos_57197559,endPos);
                                            String cellValue7 = cellStr7;
                                            dest.set(row, 7, cellValue7);
                                        }
                                        index = str.indexOf("\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",", curPos_20058273);
                                        if(index != -1) {
                                            int curPos_54788108 = index + 50;
                                            index = str.indexOf(",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"", curPos_54788108);
                                            if(index != -1) {
                                                int curPos_15575491 = index + 50;
                                                index = str.indexOf(",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"", curPos_15575491);
                                                if(index != -1) {
                                                    int curPos_50383789 = index + 50;
                                                    index = str.indexOf(",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"", curPos_50383789);
                                                    if(index != -1) {
                                                        int curPos_11954615 = index + 50;
                                                        index = str.indexOf(",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"", curPos_11954615);
                                                        if(index != -1) {
                                                            int curPos_44271891 = index + 50;
                                                            index = str.indexOf(",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"", curPos_44271891);
                                                            if(index != -1) {
                                                                int curPos_84341622 = index + 50;
                                                                index = str.indexOf(",", curPos_84341622);
                                                                if(index != -1) {
                                                                    int curPos_62678472 = index + 1;
                                                                    endPos = getEndPos(str, strLen, curPos_62678472, endWithValueString[9]);
                                                                    String cellStr9 = str.substring(curPos_62678472,endPos);
                                                                    if ( cellStr9.length() > 0 ){
                                                                        Long cellValue9;
                                                                        try{cellValue9= Long.parseLong(cellStr9); } catch(Exception e){cellValue9 = 0l;}
                                                                        if(cellValue9 != 0) {
                                                                            dest.set(row, 9, cellValue9);
                                                                            lnnz++;
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        index = str.indexOf(",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"", curPos_20058273);
                                        if(index != -1) {
                                            int curPos_89818247 = index + 50;
                                            index = str.indexOf("\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",", curPos_89818247);
                                            if(index != -1) {
                                                int curPos_51945105 = index + 50;
                                                index = str.indexOf("\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",", curPos_51945105);
                                                if(index != -1) {
                                                    int curPos_65787925 = index + 50;
                                                    index = str.indexOf(",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"", curPos_65787925);
                                                    if(index != -1) {
                                                        int curPos_67105752 = index + 50;
                                                        index = str.indexOf(",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"", curPos_67105752);
                                                        if(index != -1) {
                                                            int curPos_60302668 = index + 50;
                                                            index = str.indexOf("\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",", curPos_60302668);
                                                            if(index != -1) {
                                                                int curPos_58266363 = index + 50;
                                                                endPos = getEndPos(str, strLen, curPos_58266363, endWithValueString[8]);
                                                                String cellStr8 = str.substring(curPos_58266363,endPos);
                                                                if ( cellStr8.length() > 0 ){
                                                                    Long cellValue8;
                                                                    try{cellValue8= Long.parseLong(cellStr8); } catch(Exception e){cellValue8 = 0l;}
                                                                    if(cellValue8 != 0) {
                                                                        dest.set(row, 8, cellValue8);
                                                                        lnnz++;
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                row++;
            }}
        finally {
            IOUtilFunctions.closeSilently(reader);
        }
        return row;


    }
}
