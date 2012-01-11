package dml.runtime.matrix.io;


import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.TextInputFormat;

import dml.runtime.matrix.MetaData;


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
	public static InputInfo InputInfoForSort=new InputInfo(SequenceFileInputFormat.class, 
			DoubleWritable.class, IntWritable.class);
	public static InputInfo WeightedPairInputInfo=new InputInfo(SequenceFileInputFormat.class, 
			MatrixIndexes.class, WeightedPair.class);
	
}
