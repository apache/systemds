package dml.runtime.matrix.io;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.OutputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;

import dml.runtime.matrix.sort.CompactOutputFormat;
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
	public static OutputInfo OutputInfoForSortOutput = new OutputInfo(CompactOutputFormat.class,
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
		else if ( oi == OutputInfo.OutputInfoForSortOutput)
			return InputInfo.InputInfoForSortOutput;
		else if ( oi == OutputInfo.WeightedPairOutputInfo)
			return InputInfo.WeightedPairInputInfo;
		else 
			throw new DMLRuntimeException("Unrecognized output info: " + oi);
	}

	public static OutputInfo stringToOutputInfo (String str) {
		if ( str.equalsIgnoreCase("textcell")) {
			return TextCellOutputInfo;
		}
		else if ( str.equalsIgnoreCase("binarycell")) {
			return BinaryCellOutputInfo;
		}
		else if (str.equalsIgnoreCase("binaryblock")) {
			return BinaryBlockOutputInfo;
		}
		else if ( str.equalsIgnoreCase("sort_input") )
			return OutputInfoForSortInput;
		else if ( str.equalsIgnoreCase("sort_output"))
			return OutputInfoForSortOutput;
		else if ( str.equalsIgnoreCase("weightedpair") )
			return WeightedPairOutputInfo;
		return null;
	}
	
	public static String outputInfoToString (OutputInfo oi) {
		if ( oi == TextCellOutputInfo )
			return "textcell";
		else if ( oi == BinaryCellOutputInfo )
			return "binarycell";
		else if ( oi == BinaryBlockOutputInfo )
			return "binaryblock";
		else if ( oi == OutputInfoForSortInput )
			return "sort_input";
		else if ( oi == OutputInfoForSortOutput )
			return "sort_output";
		else if ( oi == WeightedPairOutputInfo )
			return "weightedpair";
		else
			try {
				throw new DMLRuntimeException("unrecognized outputInfo: " + oi);
			} catch (DMLRuntimeException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		return null;
	}
}