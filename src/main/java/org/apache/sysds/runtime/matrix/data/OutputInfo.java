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


package org.tugraz.sysds.runtime.matrix.data;

import static org.tugraz.sysds.parser.DataExpression.FORMAT_TYPE_VALUE_BINARY;
import static org.tugraz.sysds.parser.DataExpression.FORMAT_TYPE_VALUE_CSV;
import static org.tugraz.sysds.parser.DataExpression.FORMAT_TYPE_VALUE_LIBSVM;
import static org.tugraz.sysds.parser.DataExpression.FORMAT_TYPE_VALUE_MATRIXMARKET;
import static org.tugraz.sysds.parser.DataExpression.FORMAT_TYPE_VALUE_TEXT;

import java.io.Serializable;
import java.util.Arrays;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.OutputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.tugraz.sysds.parser.DataExpression;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.data.TensorBlock;
import org.tugraz.sysds.runtime.data.TensorIndexes;

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
	public static final OutputInfo BinaryTensorBlockOutputInfo=new OutputInfo(
			SequenceFileOutputFormat.class, TensorIndexes.class, TensorBlock.class);
	public static final OutputInfo BinaryBlockFrameOutputInfo=new OutputInfo(
			SequenceFileOutputFormat.class, LongWritable.class, FrameBlock.class);

	public static final OutputInfo CSVOutputInfo = null;
	public static final OutputInfo LIBSVMOutputInfo = new OutputInfo (TextOutputFormat.class, 
			NullWritable.class, Text.class);
	
	public static InputInfo getMatchingInputInfo(OutputInfo oi) {
		if ( oi == OutputInfo.BinaryBlockOutputInfo )
			return InputInfo.BinaryBlockInputInfo;
		else if ( oi == OutputInfo.BinaryTensorBlockOutputInfo )
			return InputInfo.BinaryTensorBlockInputInfo;
		else if ( oi == OutputInfo.MatrixMarketOutputInfo )
			return InputInfo.MatrixMarketInputInfo;
		else if ( oi == OutputInfo.BinaryCellOutputInfo ) 
			return InputInfo.BinaryCellInputInfo;
		else if ( oi == OutputInfo.TextCellOutputInfo )
			return InputInfo.TextCellInputInfo;
		else if ( oi == OutputInfo.CSVOutputInfo)
			return InputInfo.CSVInputInfo;
		else if ( oi == OutputInfo.LIBSVMOutputInfo)
			return InputInfo.LIBSVMInputInfo;
		else 
			throw new DMLRuntimeException("Unrecognized output info: " + oi);
	}
		
	public static OutputInfo stringToOutputInfo (String str) {
		if ( str.equalsIgnoreCase("textcell"))
			return TextCellOutputInfo;
		if ( str.equalsIgnoreCase("text"))
			return TextCellOutputInfo;
		else if ( str.equalsIgnoreCase("matrixmarket"))
			return MatrixMarketOutputInfo;
		else if ( str.equalsIgnoreCase("mm"))
			return MatrixMarketOutputInfo;
		else if ( str.equalsIgnoreCase("binarycell"))
			return BinaryCellOutputInfo;
		else if (str.equalsIgnoreCase("binaryblock"))
			return BinaryBlockOutputInfo;
		else if (str.equalsIgnoreCase("binary"))
			return BinaryBlockOutputInfo;
		else if (str.equalsIgnoreCase("binarytensorblock"))
			return BinaryTensorBlockOutputInfo;
		else if ( str.equalsIgnoreCase("csv") )
			return CSVOutputInfo;
		else if ( str.equalsIgnoreCase("libsvm") )
			return LIBSVMOutputInfo;
		return null;
	}
	
	public static String outputInfoToString (OutputInfo oi) {
		if ( oi == TextCellOutputInfo )
			return "textcell";
		else if ( oi == MatrixMarketOutputInfo)
			return "matrixmarket";
		else if ( oi == BinaryCellOutputInfo )
			return "binarycell";
		else if ( oi == BinaryBlockOutputInfo )
			return "binaryblock";
		else if ( oi == BinaryTensorBlockOutputInfo )
			return "binarytensorblock";
		else if ( oi == CSVOutputInfo )
			return "csv";
		else if ( oi == LIBSVMOutputInfo)
			return "libsvm";
		else
			throw new DMLRuntimeException("Unrecognized outputInfo: " + oi);
	}

	public static String outputInfoToStringExternal(OutputInfo oinfo) 
	{
		if( oinfo == OutputInfo.TextCellOutputInfo )
			return DataExpression.FORMAT_TYPE_VALUE_TEXT;
		else if( oinfo == OutputInfo.MatrixMarketOutputInfo )
			return DataExpression.FORMAT_TYPE_VALUE_MATRIXMARKET;
		else if( oinfo == OutputInfo.CSVOutputInfo )
			return DataExpression.FORMAT_TYPE_VALUE_CSV;
		else if( oinfo == OutputInfo.LIBSVMOutputInfo)
			return DataExpression.FORMAT_TYPE_VALUE_LIBSVM;
		else if( oinfo == OutputInfo.BinaryBlockOutputInfo 
				|| oinfo == OutputInfo.BinaryCellOutputInfo
				|| oinfo == OutputInfo.BinaryTensorBlockOutputInfo)
			return DataExpression.FORMAT_TYPE_VALUE_BINARY;
		else
			return "specialized";
	}
	
	public static OutputInfo outputInfoFromStringExternal(String format) {
		if (format.equalsIgnoreCase(FORMAT_TYPE_VALUE_TEXT))
			return OutputInfo.TextCellOutputInfo;
		else if (format.equalsIgnoreCase(FORMAT_TYPE_VALUE_BINARY))
			return OutputInfo.BinaryBlockOutputInfo;
		else if (format.equalsIgnoreCase(FORMAT_TYPE_VALUE_MATRIXMARKET))
			return OutputInfo.MatrixMarketOutputInfo;
		else if (format.equalsIgnoreCase(FORMAT_TYPE_VALUE_LIBSVM))
			return OutputInfo.LIBSVMOutputInfo;
		else if (format.equalsIgnoreCase(FORMAT_TYPE_VALUE_CSV))
			return OutputInfo.CSVOutputInfo;
		throw new DMLRuntimeException("Invalid external file format: "+format);
	}
	
	@Override 
	public int hashCode() {
		return Arrays.hashCode(new int[] {
			outputFormatClass.hashCode(),
			outputKeyClass.hashCode(),
			outputValueClass.hashCode()
		});
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
