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

package org.apache.sysds.runtime.util;

import java.io.IOException;

import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.hadoop.mapred.FileSplit;

/**
 * Custom binary block input format to return the custom record reader.
 * <p>
 * NOTE: Not used by default.
 * <p>
 * NOTE: Used for performance debugging of binary block HDFS reads.
 */
public class BinaryBlockInputFormat extends SequenceFileInputFormat<MatrixIndexes,MatrixBlock>
{
	@Override
	public RecordReader<MatrixIndexes, MatrixBlock> getRecordReader(InputSplit split, JobConf job, Reporter reporter)
		throws IOException 
	{
		return new BinaryBlockRecordReader(job, (FileSplit)split);
	}
}