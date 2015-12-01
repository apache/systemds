/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */


package org.apache.sysml.runtime.matrix.data;


import java.io.Serializable;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.TextInputFormat;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.MetaData;
import org.apache.sysml.runtime.matrix.sort.PickFromCompactInputFormat;

@SuppressWarnings("rawtypes")
public class InputInfo implements Serializable 
{

	private static final long serialVersionUID = 7059677437144672023L;

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
	public static final InputInfo TextCellInputInfo=new InputInfo(TextInputFormat.class, 
			 LongWritable.class, Text.class);
	public static final InputInfo MatrixMarketInputInfo = new InputInfo (TextInputFormat.class, 
			 LongWritable.class, Text.class);
	public static final InputInfo BinaryCellInputInfo=new InputInfo(SequenceFileInputFormat.class, 
			MatrixIndexes.class, MatrixCell.class);
	public static final InputInfo BinaryBlockInputInfo=new InputInfo(
			//for jobs like GMR, we use CombineSequenceFileInputFormat (which requires to specify the maxsplitsize, hence not included here)
			SequenceFileInputFormat.class, MatrixIndexes.class, MatrixBlock.class); 
	
	// Format that denotes the input of a SORT job
	public static final InputInfo InputInfoForSort=new InputInfo(SequenceFileInputFormat.class, 
			DoubleWritable.class, IntWritable.class);
	
	// Format that denotes the output of a SORT job
	public static final InputInfo InputInfoForSortOutput = new InputInfo(PickFromCompactInputFormat.class,
			DoubleWritable.class, IntWritable.class);

	public static final InputInfo WeightedPairInputInfo=new InputInfo(SequenceFileInputFormat.class, 
			MatrixIndexes.class, WeightedPair.class);
	
	public static final InputInfo CSVInputInfo=new InputInfo(TextInputFormat.class, 
			 LongWritable.class, Text.class);
	
	public static OutputInfo getMatchingOutputInfo(InputInfo ii) throws DMLRuntimeException {
		if ( ii == InputInfo.BinaryBlockInputInfo )
			return OutputInfo.BinaryBlockOutputInfo;
		else if ( ii == InputInfo.MatrixMarketInputInfo)
			return OutputInfo.MatrixMarketOutputInfo;
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
		else if ( ii == InputInfo.CSVInputInfo)
			return OutputInfo.CSVOutputInfo;
		else 
			throw new DMLRuntimeException("Unrecognized output info: " + ii);
	}
	
	public static InputInfo stringToInputInfo (String str) {
		if ( str.equalsIgnoreCase("textcell")) {
			return TextCellInputInfo;
		}
		if ( str.equalsIgnoreCase("matrixmarket")) {
			return MatrixMarketInputInfo;
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
		else if ( str.equalsIgnoreCase("csv"))
			return CSVInputInfo;
		return null;
	}
	
	public static String inputInfoToString (InputInfo ii) 
		throws DMLRuntimeException 
	{
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
		else if ( ii == MatrixMarketInputInfo )
			return "matrixmarket";
		else if ( ii == CSVInputInfo )
			return "csv";
		else
			throw new DMLRuntimeException("Unrecognized inputInfo: " + ii);
	}
	
	
}
