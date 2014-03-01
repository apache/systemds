/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Iterator;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.controlprogram.parfor.util.PairWritableBlock;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.PairWritableCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;

/**
 * Remote data partitioner reducer implementation that takes the given
 * partitions and writes it to individual files. The reasons for the design 
 * with a reducer instead of MultipleOutputs are 
 * 1) robustness wrt num open file descriptors, outofmem
 * 2) no append for sequence files 
 * 3) performance of actual write
 *
 */
public class DataPartitionerRemoteReducer 
	implements Reducer<LongWritable, Writable, Writable, Writable>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private DataPartitionerReducer _reducer = null; 
	
	public DataPartitionerRemoteReducer( ) 
	{
		
	}
	
	@Override
	public void reduce(LongWritable key, Iterator<Writable> valueList, OutputCollector<Writable, Writable> out, Reporter reporter)
		throws IOException 
	{
		_reducer.processKeyValueList(key, valueList, out, reporter);
	}

	/**
	 * 
	 */
	public void configure(JobConf job)
	{
		String fnameNew = MRJobConfiguration.getPartitioningFilename( job );
		OutputInfo oi = MRJobConfiguration.getPartitioningOutputInfo( job );
		
		if( oi == OutputInfo.TextCellOutputInfo )
			_reducer = new DataPartitionerReducerTextcell(job, fnameNew);
		else if( oi == OutputInfo.BinaryCellOutputInfo )
			_reducer = new DataPartitionerReducerBinarycell(job, fnameNew);
		else if( oi == OutputInfo.BinaryBlockOutputInfo )
			_reducer = new DataPartitionerReducerBinaryblock(job, fnameNew);
		else
			throw new RuntimeException("Unable to configure reducer with unknown output info: "+oi.toString());	
	}
	
	/**
	 * 
	 */
	@Override
	public void close() throws IOException 
	{
		//do nothing
	}

	
	private abstract class DataPartitionerReducer //NOTE: could also be refactored as three different reducers
	{
		protected JobConf _job = null;
		protected FileSystem _fs = null;
		protected String _fnameNew = null;
		
		protected DataPartitionerReducer( JobConf job, String fnameNew ) 
		{
			_job = job;
			_fnameNew = fnameNew;
			
			try {
				_fs = FileSystem.get(_job);
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
		
		protected abstract void processKeyValueList( LongWritable key, Iterator<Writable> valueList, OutputCollector<Writable, Writable> out, Reporter reporter ) 
			throws IOException;
	}

	private class DataPartitionerReducerTextcell extends DataPartitionerReducer
	{
		private StringBuilder _sb = null;
		
		protected DataPartitionerReducerTextcell( JobConf job, String fnameNew )
		{
			super(job, fnameNew);
			
			//for obj reuse and preventing repeated buffer re-allocations
			_sb = new StringBuilder();
		}

		@Override
		protected void processKeyValueList(LongWritable key, Iterator<Writable> valueList, OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException 
		{
			
			BufferedWriter writer = null;
			try
			{			
				Path path = new Path(_fnameNew+"/"+key.get());
				writer = new BufferedWriter(new OutputStreamWriter(_fs.create(path,true)));		
		        
				while( valueList.hasNext() )
				{
					PairWritableCell pairValue = (PairWritableCell)valueList.next();
					
					_sb.append(pairValue.indexes.getRowIndex());
					_sb.append(' ');
					_sb.append(pairValue.indexes.getColumnIndex());
					_sb.append(' ');
					_sb.append(pairValue.cell.getValue());
					_sb.append('\n');	
					writer.write(_sb.toString());
					
					_sb.setLength(0);
				}
			} 
			finally
			{
				if( writer != null )
					writer.close();
			}
		}
		
	}
	

	private class DataPartitionerReducerBinarycell extends DataPartitionerReducer
	{
		protected DataPartitionerReducerBinarycell( JobConf job, String fnameNew )
		{
			super(job, fnameNew);
		}

		@Override
		protected void processKeyValueList(LongWritable key, Iterator<Writable> valueList, OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException 
		{
			SequenceFile.Writer writer = null;
			try
			{			
				Path path = new Path(_fnameNew+"/"+key.get());
				writer = new SequenceFile.Writer(_fs, _job, path, MatrixIndexes.class, MatrixCell.class);
				while( valueList.hasNext() )
				{
					PairWritableCell pairValue = (PairWritableCell)valueList.next();
					writer.append(pairValue.indexes, pairValue.cell);
				}
			} 
			finally
			{
				if( writer != null )
					writer.close();
			}
		}
		
	}
	
	private class DataPartitionerReducerBinaryblock extends DataPartitionerReducer
	{
		protected DataPartitionerReducerBinaryblock( JobConf job, String fnameNew )
		{
			super(job, fnameNew);
		}

		@Override
		protected void processKeyValueList(LongWritable key, Iterator<Writable> valueList, OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException 
		{
			SequenceFile.Writer writer = null;
			try
			{			
				Path path = new Path(_fnameNew+"/"+key.get());
				writer = new SequenceFile.Writer(_fs, _job, path, MatrixIndexes.class, MatrixBlock.class);
				while( valueList.hasNext() )
				{
					PairWritableBlock pairValue = (PairWritableBlock)valueList.next();
					writer.append(pairValue.indexes, pairValue.block);
				}
			} 
			finally
			{
				if( writer != null )
					writer.close();
			}
		}
		
	}
}
