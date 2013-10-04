/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.TaggedMatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.TaggedMatrixCell;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;

/**
 * Remote resultmerge mapper implementation that does the preprocessing
 * in terms of tagging .
 *
 */
public class ResultMergeRemoteMapper 
	implements Mapper<Writable, Writable, Writable, Writable>
{		
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private ResultMergeMapper _mapper;
	
	
	public void map(Writable key, Writable value, OutputCollector<Writable, Writable> out, Reporter reporter) 
		throws IOException
	{
		_mapper.processKeyValue(key, value, out, reporter);	
	}

	public void configure(JobConf job)
	{
		InputInfo ii = MRJobConfiguration.getResultMergeInputInfo(job);
		String compareFname = MRJobConfiguration.getResultMergeInfoCompareFilename(job);
		String currentFname = job.get("map.input.file");
		
		byte tag = 0;
		//startsWith comparison in order to account for part names in currentFname
		if( currentFname.startsWith(compareFname) ) 
			tag = ResultMergeRemoteMR.COMPARE_TAG;
		else
			tag = ResultMergeRemoteMR.DATA_TAG;
		
		if( ii == InputInfo.TextCellInputInfo )
			_mapper = new ResultMergeMapperTextCell(tag);
		else if( ii == InputInfo.BinaryCellInputInfo )
			_mapper = new ResultMergeMapperBinaryCell(tag);
		else if( ii == InputInfo.BinaryBlockInputInfo )
			_mapper = new ResultMergeMapperBinaryBlock(tag);
		else
			throw new RuntimeException("Unable to configure mapper with unknown input info: "+ii.toString());
	}
	
	/**
	 * 
	 */
	@Override
	public void close() throws IOException 
	{
		//do nothing
	}
	
	private abstract class ResultMergeMapper
	{
		protected byte _tag = 0;
		
		protected ResultMergeMapper( byte tag )
		{
			_tag = tag;
		}
		
		protected abstract void processKeyValue(Writable key, Writable value, OutputCollector<Writable, Writable> out, Reporter reporter) 
			throws IOException;	
	}
	
	protected class ResultMergeMapperTextCell extends ResultMergeMapper
	{
		private MatrixIndexes _objKey;
		private MatrixCell _objValueHelp;
		private TaggedMatrixCell _objValue;
		
		protected ResultMergeMapperTextCell(byte tag)
		{
			super(tag);
			_objKey = new MatrixIndexes();
			_objValueHelp = new MatrixCell();
			_objValue = new TaggedMatrixCell();
			_objValue.setTag( _tag );
		}

		@Override
		protected void processKeyValue(Writable key, Writable value, OutputCollector<Writable, Writable> out, Reporter reporter)
				throws IOException 
		{
			String cellStr = ((Text)value).toString().trim();	
			StringTokenizer st = new StringTokenizer(cellStr, " ");
			long row = Long.parseLong( st.nextToken() );
			long col = Long.parseLong( st.nextToken() );
			double lvalue = Double.parseDouble( st.nextToken() );
			
			_objKey.setIndexes(row,col);
			_objValueHelp.setValue(lvalue);
			_objValue.setBaseObject(_objValueHelp);
			
			out.collect(_objKey, _objValue);
		}	
	}
	
	protected class ResultMergeMapperBinaryCell extends ResultMergeMapper
	{
		private TaggedMatrixCell _objValue;
		
		protected ResultMergeMapperBinaryCell(byte tag)
		{
			super(tag);
			_objValue = new TaggedMatrixCell();
			_objValue.setTag( _tag );
		}

		@Override
		protected void processKeyValue(Writable key, Writable value, OutputCollector<Writable, Writable> out, Reporter reporter)
				throws IOException 
		{
			_objValue.setBaseObject((MatrixCell)value);
			out.collect(key, _objValue);
		}	
	}
	
	protected class ResultMergeMapperBinaryBlock extends ResultMergeMapper
	{
		private TaggedMatrixBlock _objValue;
		
		protected ResultMergeMapperBinaryBlock(byte tag)
		{
			super(tag);
			_objValue = new TaggedMatrixBlock();
			_objValue.setTag( _tag );
		}

		@Override
		protected void processKeyValue(Writable key, Writable value, OutputCollector<Writable, Writable> out, Reporter reporter)
				throws IOException 
		{
			_objValue.setBaseObject((MatrixBlock)value);
			out.collect(key, _objValue);
		}	
	}
}
