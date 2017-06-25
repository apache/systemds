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

package org.apache.sysml.runtime.controlprogram.parfor;

import java.io.File;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.spark.api.java.function.VoidFunction;

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.controlprogram.parfor.util.PairWritableBlock;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.mapred.MRConfigurationNames;

import scala.Tuple2;

public class DataPartitionerRemoteSparkReducer implements VoidFunction<Tuple2<Long, Iterable<Writable>>> 
{
	private static final long serialVersionUID = -7149865018683261964L;
	
	private final String _fnameNew;
	private final int _replication;
	
	public DataPartitionerRemoteSparkReducer(String fnameNew, OutputInfo oi, int replication) {
		_fnameNew = fnameNew;
		_replication = replication;
	}

	@Override
	@SuppressWarnings("deprecation")	
	public void call(Tuple2<Long, Iterable<Writable>> arg0)
		throws Exception 
	{
		//prepare grouped partition input
		Long key = arg0._1();
		Iterator<Writable> valueList = arg0._2().iterator();
		
		//write entire partition to binary block sequence file
		SequenceFile.Writer writer = null;
		try
		{
			//create sequence file writer
			Configuration job = new Configuration(ConfigurationManager.getCachedJobConf());
			Path path = new Path(_fnameNew + File.separator + key);
			FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
			writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class, 
					job.getInt(MRConfigurationNames.IO_FILE_BUFFER_SIZE, 4096),
                    (short)_replication, fs.getDefaultBlockSize(), null, new SequenceFile.Metadata());	
			
			//write individual blocks unordered to output
			while( valueList.hasNext() ) {
				PairWritableBlock pair = (PairWritableBlock) valueList.next();
				writer.append(pair.indexes, pair.block);
			}
		} 
		finally {
			IOUtilFunctions.closeSilently(writer);
		}	
	}	
}
