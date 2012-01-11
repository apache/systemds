package dml.runtime.matrix.io;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.OutputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;

import dml.utils.DMLRuntimeException;



public class OutputInfo {

	public Class<? extends OutputFormat> outputFormatClass;
	public Class<? extends Writable> outputKeyClass;
	public Class<? extends Writable> outputValueClass;
	public OutputInfo(Class<? extends OutputFormat> formatCls,
			Class<? extends Writable> keyCls, Class<? extends Writable> valueCls)
	{
		outputFormatClass=formatCls;
		outputKeyClass=keyCls;
		outputValueClass=valueCls;
	}
	public static OutputInfo TextCellOutputInfo=new OutputInfo(TextOutputFormat.class, 
			NullWritable.class, Text.class);
	public static OutputInfo BinaryCellOutputInfo=new OutputInfo(SequenceFileOutputFormat.class, 
			MatrixIndexes.class, MatrixCell.class);
//	public static OutputInfo TextBlockOutputInfo=new OutputInfo(TextOutputFormat.class, 
//			NullWritable.class, Text.class);
	public static OutputInfo BinaryBlockOutputInfo=new OutputInfo(SequenceFileOutputFormat.class, 
			MatrixIndexes.class, MatrixBlock.class);
	public static OutputInfo OutputInfoForSortInput=new OutputInfo(SequenceFileOutputFormat.class, 
			DoubleWritable.class, IntWritable.class);
	public static OutputInfo WeightedPairOutputInfo=new OutputInfo(SequenceFileOutputFormat.class, 
			MatrixIndexes.class, WeightedPair.class);

	public static InputInfo getMatchingInputInfo(OutputInfo oi) throws DMLRuntimeException {
		if ( oi == OutputInfo.BinaryBlockOutputInfo )
			return InputInfo.BinaryBlockInputInfo;
		else if ( oi == OutputInfo.BinaryCellOutputInfo ) 
			return InputInfo.BinaryCellInputInfo;
		else if ( oi == OutputInfo.TextCellOutputInfo )
			return InputInfo.TextCellInputInfo;
		else if ( oi == OutputInfo.OutputInfoForSortInput)
			return InputInfo.InputInfoForSort;
		else if ( oi == OutputInfo.WeightedPairOutputInfo)
			return InputInfo.WeightedPairInputInfo;
		else 
			throw new DMLRuntimeException("Unrecognized output info: " + oi);
	}
}