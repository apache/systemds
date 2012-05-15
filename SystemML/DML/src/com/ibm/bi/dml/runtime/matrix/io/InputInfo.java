package com.ibm.bi.dml.runtime.matrix.io;


import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.TextInputFormat;

import com.ibm.bi.dml.runtime.matrix.MetaData;
import com.ibm.bi.dml.runtime.matrix.sort.CompactInputFormat;
import com.ibm.bi.dml.utils.DMLRuntimeException;



public class InputInfo {

	public Class<? extends InputFormat> inputFormatClass;
	public Class<? extends Writable> inputKeyClass;
	public Class<? extends Writable> inputValueClass;
	public MetaData metadata=null;
	public InputInfo(Class<? extends InputFormat> formatCls, 
			Class<? extends Writable> keyCls, Class<? extends Writable> valueCls)
	{
		inputFormatClass=formatCls;
		inputKeyClass=keyCls;
		inputValueClass=valueCls;
	}
	public InputInfo(Class<? extends InputFormat> formatCls, 
			Class<? extends Writable> keyCls, Class<? extends Writable> valueCls, MetaData md)
	{
		inputFormatClass=formatCls;
		inputKeyClass=keyCls;
		inputValueClass=valueCls;
		metadata=md;
	}
	public void setMetaData(MetaData md)
	{
		metadata=md;
	}
	public static InputInfo TextCellInputInfo=new InputInfo(TextInputFormat.class, 
			 LongWritable.class, Text.class);
	public static InputInfo BinaryCellInputInfo=new InputInfo(SequenceFileInputFormat.class, 
			MatrixIndexes.class, MatrixCell.class);
	//public static InputInfo TextBlockInputInfo=new InputInfo(TextInputFormat.class, 
	//		LongWritable.class, Text.class);
	public static InputInfo BinaryBlockInputInfo=new InputInfo(SequenceFileInputFormat.class, 
			MatrixIndexes.class, MatrixBlock.class);
	
	// Format that denotes the input of a SORT job
	public static InputInfo InputInfoForSort=new InputInfo(SequenceFileInputFormat.class, 
			DoubleWritable.class, IntWritable.class);
	
	// Format that denotes the output of a SORT job
	public static InputInfo InputInfoForSortOutput = new InputInfo(CompactInputFormat.class,
			DoubleWritable.class, IntWritable.class);

	public static InputInfo WeightedPairInputInfo=new InputInfo(SequenceFileInputFormat.class, 
			MatrixIndexes.class, WeightedPair.class);
	
	public static OutputInfo getMatchingOutputInfo(InputInfo ii) throws DMLRuntimeException {
		if ( ii == InputInfo.BinaryBlockInputInfo )
			return OutputInfo.BinaryBlockOutputInfo;
		else if ( ii == InputInfo.BinaryCellInputInfo ) 
			return OutputInfo.BinaryCellOutputInfo;
		else if ( ii == InputInfo.TextCellInputInfo )
			return OutputInfo.TextCellOutputInfo;
		else if ( ii == InputInfo.InputInfoForSort)
			return OutputInfo.OutputInfoForSortInput;
		else if ( ii == InputInfo.InputInfoForSortOutput)
			return OutputInfo.OutputInfoForSortOutput;
		else if ( ii == InputInfo.WeightedPairInputInfo)
			return OutputInfo.WeightedPairOutputInfo;
		else 
			throw new DMLRuntimeException("Unrecognized output info: " + ii);
	}
	
	public static InputInfo stringToInputInfo (String str) {
		if ( str.equalsIgnoreCase("textcell")) {
			return TextCellInputInfo;
		}
		else if ( str.equalsIgnoreCase("binarycell")) {
			return BinaryCellInputInfo;
		}
		else if (str.equalsIgnoreCase("binaryblock")) {
			return BinaryBlockInputInfo;
		}
		else if ( str.equalsIgnoreCase("sort_input"))
			return InputInfoForSort;
		else if ( str.equalsIgnoreCase("sort_output"))
			return InputInfoForSortOutput;
		else if ( str.equalsIgnoreCase("weightedpair"))
			return WeightedPairInputInfo;
		return null;
	}
	
	public static String inputInfoToString (InputInfo ii) {
		if ( ii == TextCellInputInfo )
			return "textcell";
		else if ( ii == BinaryCellInputInfo )
			return "binarycell";
		else if ( ii == BinaryBlockInputInfo )
			return "binaryblock";
		else if ( ii == InputInfoForSort )
			return "sort_input";
		else if ( ii == InputInfoForSortOutput)
			return "sort_output";
		else if ( ii == WeightedPairInputInfo )
			return "weightedpair";
		return null;
	}
	
	
}
