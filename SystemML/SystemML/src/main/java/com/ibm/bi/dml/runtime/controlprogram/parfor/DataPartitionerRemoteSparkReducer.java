/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
