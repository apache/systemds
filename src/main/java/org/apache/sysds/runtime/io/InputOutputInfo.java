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

package org.apache.sysds.runtime.io;

import java.io.Serializable;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.OutputFormat;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.data.TensorIndexes;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.meta.MetaData;

@SuppressWarnings("rawtypes")
public class InputOutputInfo implements Serializable
{
	private static final long serialVersionUID = 7059677437144672023L;

	public Class<? extends InputFormat> inputFormatClass;
	public Class<? extends OutputFormat> outputFormatClass;
	public Class<? extends Writable> keyClass;
	public Class<? extends Writable> valueClass;
	public MetaData metadata=null;
	
	public InputOutputInfo(Class<? extends InputFormat> formatClsIn, Class<? extends OutputFormat> formatClsOut,
		Class<? extends Writable> keyCls, Class<? extends Writable> valueCls)
	{
		inputFormatClass = formatClsIn;
		outputFormatClass = formatClsOut;
		keyClass = keyCls;
		valueClass = valueCls;
	}

	public static final InputOutputInfo TextCellInputOutputInfo = new InputOutputInfo(
		TextInputFormat.class, TextOutputFormat.class, LongWritable.class, Text.class);
	public static final InputOutputInfo MatrixMarketInputOutputInfo = new InputOutputInfo(
		TextInputFormat.class, TextOutputFormat.class, LongWritable.class, Text.class);
	public static final InputOutputInfo BinaryBlockInputOutputInfo = new InputOutputInfo(
		SequenceFileInputFormat.class, SequenceFileOutputFormat.class, MatrixIndexes.class, MatrixBlock.class);
	public static final InputOutputInfo BinaryBlockFrameInputOutputInfo = new InputOutputInfo(
		SequenceFileInputFormat.class, SequenceFileOutputFormat.class, LongWritable.class, FrameBlock.class);
	public static final InputOutputInfo BinaryBlockTensorInputOutputInfo = new InputOutputInfo(
		SequenceFileInputFormat.class, SequenceFileOutputFormat.class, TensorIndexes.class, TensorBlock.class);
	public static final InputOutputInfo CSVInputOutputInfo = new InputOutputInfo(
		TextInputFormat.class, TextOutputFormat.class, LongWritable.class, Text.class);
	public static final InputOutputInfo LIBSVMInputOutputInfo = new InputOutputInfo(
		TextInputFormat.class, TextOutputFormat.class, LongWritable.class, Text.class);

	@SuppressWarnings("incomplete-switch")
	public static InputOutputInfo get(DataType dt, FileFormat fmt) {
		switch(fmt) {
			case TEXT:   return TextCellInputOutputInfo;
			case MM:     return MatrixMarketInputOutputInfo;
			case CSV:    return CSVInputOutputInfo;
			case LIBSVM: return LIBSVMInputOutputInfo;
			case BINARY: {
				switch( dt ) {
					case MATRIX: return BinaryBlockInputOutputInfo;
					case FRAME:  return BinaryBlockFrameInputOutputInfo;
					case TENSOR: return BinaryBlockTensorInputOutputInfo;
				}
			}
		}
		throw new DMLRuntimeException(
			"Could not obtain input/output info for format: " + dt.toString() + ", " + fmt.toString());
	}
}
