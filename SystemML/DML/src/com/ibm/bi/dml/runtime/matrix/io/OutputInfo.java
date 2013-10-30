/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.io;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.OutputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.WriteCSVMR.RowBlockForTextOutput;
import com.ibm.bi.dml.runtime.matrix.sort.CompactOutputFormat;




public class OutputInfo 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
	public static OutputInfo MatrixMarketOutputInfo  = new OutputInfo (TextOutputFormat.class, 
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
	public static OutputInfo CSVOutputInfo=new OutputInfo(UnPaddedOutputFormat.class, 
			NullWritable.class, RowBlockForTextOutput.class);

	public static InputInfo getMatchingInputInfo(OutputInfo oi) throws DMLRuntimeException {
		if ( oi == OutputInfo.BinaryBlockOutputInfo )
			return InputInfo.BinaryBlockInputInfo;
		else if ( oi == OutputInfo.MatrixMarketOutputInfo )
			return InputInfo.MatrixMarketInputInfo;
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
		else if ( oi == OutputInfo.CSVOutputInfo)
			return InputInfo.CSVInputInfo;
		else 
			throw new DMLRuntimeException("Unrecognized output info: " + oi);
	}

	public static OutputInfo stringToOutputInfo (String str) {
		if ( str.equalsIgnoreCase("textcell")) {
			return TextCellOutputInfo;
		}
		else if ( str.equalsIgnoreCase("matrixmarket")) {
			return MatrixMarketOutputInfo;
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
		else if ( str.equalsIgnoreCase("csv") )
			return CSVOutputInfo;
		return null;
	}
	
	public static String outputInfoToString (OutputInfo oi) 
		throws DMLRuntimeException
	{
		if ( oi == TextCellOutputInfo )
			return "textcell";
		else if ( oi == MatrixMarketOutputInfo)
			return "matrixmarket";
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
		else if ( oi == CSVOutputInfo )
			return "csv";
		else
			throw new DMLRuntimeException("Unrecognized outputInfo: " + oi);
	}
}
