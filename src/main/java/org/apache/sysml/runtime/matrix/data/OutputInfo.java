/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */


package org.apache.sysml.runtime.matrix.data;

import java.io.Serializable;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.OutputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.sysml.parser.DataExpression;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.mapred.CSVWriteReducer.RowBlockForTextOutput;
import org.apache.sysml.runtime.matrix.sort.CompactOutputFormat;



@SuppressWarnings("rawtypes")
public class OutputInfo implements Serializable 
{

	private static final long serialVersionUID = -3115943514779675817L;

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
	public static final OutputInfo TextCellOutputInfo=new OutputInfo(TextOutputFormat.class, 
			NullWritable.class, Text.class);
	public static final OutputInfo MatrixMarketOutputInfo  = new OutputInfo (TextOutputFormat.class, 
			NullWritable.class, Text.class);
	public static final OutputInfo BinaryCellOutputInfo=new OutputInfo(SequenceFileOutputFormat.class, 
			MatrixIndexes.class, MatrixCell.class);
	public static final OutputInfo BinaryBlockOutputInfo=new OutputInfo(
			SequenceFileOutputFormat.class, MatrixIndexes.class, MatrixBlock.class);
	public static final OutputInfo BinaryBlockFrameOutputInfo=new OutputInfo(
			SequenceFileOutputFormat.class, LongWritable.class, FrameBlock.class);
	public static final OutputInfo OutputInfoForSortInput=new OutputInfo(SequenceFileOutputFormat.class, 
			DoubleWritable.class, IntWritable.class);
	public static final OutputInfo OutputInfoForSortOutput = new OutputInfo(CompactOutputFormat.class,
			DoubleWritable.class, IntWritable.class);
	public static final OutputInfo WeightedPairOutputInfo=new OutputInfo(SequenceFileOutputFormat.class, 
			MatrixIndexes.class, WeightedPair.class);
	public static final OutputInfo CSVOutputInfo=new OutputInfo(UnPaddedOutputFormat.class, 
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
	
	/**
	 * 
	 * @param oinfo
	 * @return
	 */
	public static String outputInfoToStringExternal(OutputInfo oinfo) 
	{
		if( oinfo == OutputInfo.TextCellOutputInfo )
			return DataExpression.FORMAT_TYPE_VALUE_TEXT;
		else if( oinfo == OutputInfo.MatrixMarketOutputInfo )
			return DataExpression.FORMAT_TYPE_VALUE_MATRIXMARKET;
		else if( oinfo == OutputInfo.CSVOutputInfo )
			return DataExpression.FORMAT_TYPE_VALUE_CSV;
		else if( oinfo == OutputInfo.BinaryBlockOutputInfo 
				|| oinfo == OutputInfo.BinaryCellOutputInfo )
			return DataExpression.FORMAT_TYPE_VALUE_BINARY;
		else
			return "specialized";
	}
	
	@Override
	public boolean equals( Object o ) 
	{
		if( !(o instanceof OutputInfo) )
			return false;
		
		OutputInfo that = (OutputInfo) o;
		return ( outputFormatClass == that.outputFormatClass
			 	&& outputKeyClass == that.outputKeyClass
			 	&& outputValueClass == that.outputValueClass );
	}
}
