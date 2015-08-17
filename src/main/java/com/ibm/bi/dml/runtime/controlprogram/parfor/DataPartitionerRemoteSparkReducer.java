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

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.File;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.spark.api.java.function.VoidFunction;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.PairWritableBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;

import scala.Tuple2;

/**
 * 
 */
public class DataPartitionerRemoteSparkReducer implements VoidFunction<Tuple2<Long, Iterable<Writable>>> 
{
	
	private static final long serialVersionUID = -7149865018683261964L;
	
	private String _fnameNew = null;
	
	public DataPartitionerRemoteSparkReducer(String fnameNew, OutputInfo oi) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		_fnameNew = fnameNew;
		//_oi = oi;
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
			Configuration job = new Configuration();
			FileSystem fs = FileSystem.get(job);
			Path path = new Path(_fnameNew + File.separator + key);
			writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class);
			while( valueList.hasNext() )
			{
				PairWritableBlock pair = (PairWritableBlock) valueList.next();
				writer.append(pair.indexes, pair.block);
			}
		} 
		finally
		{
			if( writer != null )
				writer.close();
		}	
	}
	
}
