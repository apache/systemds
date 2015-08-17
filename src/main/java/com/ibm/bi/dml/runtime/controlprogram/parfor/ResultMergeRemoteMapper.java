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

import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.TaggedMatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.TaggedMatrixCell;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.FastStringTokenizer;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

/**
 * Remote resultmerge mapper implementation that does the preprocessing
 * in terms of tagging .
 *
 */
public class ResultMergeRemoteMapper 
	implements Mapper<Writable, Writable, Writable, Writable>
{		
	
	private ResultMergeMapper _mapper;
	
	public void map(Writable key, Writable value, OutputCollector<Writable, Writable> out, Reporter reporter) 
		throws IOException
	{
		//tag and pass-through matrix values 
		_mapper.processKeyValue(key, value, out, reporter);	
	}

	public void configure(JobConf job)
	{
		InputInfo ii = MRJobConfiguration.getResultMergeInputInfo(job);
		long[] tmp = MRJobConfiguration.getResultMergeMatrixCharacteristics( job );
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
			_mapper = new ResultMergeMapperBinaryBlock(tag, tmp[0], tmp[1], tmp[2], tmp[3]);
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
	
	private static abstract class ResultMergeMapper
	{
		protected byte _tag = 0;
		
		protected ResultMergeMapper( byte tag )
		{
			_tag = tag;
		}
		
		protected abstract void processKeyValue(Writable key, Writable value, OutputCollector<Writable, Writable> out, Reporter reporter) 
			throws IOException;	
	}
	
	protected static class ResultMergeMapperTextCell extends ResultMergeMapper
	{
		private MatrixIndexes _objKey;
		private MatrixCell _objValueHelp;
		private TaggedMatrixCell _objValue;
		private FastStringTokenizer _st;
		
		protected ResultMergeMapperTextCell(byte tag)
		{
			super(tag);
			_objKey = new MatrixIndexes();
			_objValueHelp = new MatrixCell();
			_objValue = new TaggedMatrixCell();
			_objValue.setTag( _tag );
			
			_st = new FastStringTokenizer(' ');
		}

		@Override
		protected void processKeyValue(Writable key, Writable value, OutputCollector<Writable, Writable> out, Reporter reporter)
				throws IOException 
		{
			_st.reset( value.toString() ); //reset tokenizer
			long row = _st.nextLong();
			long col = _st.nextLong();
			double lvalue = _st.nextDouble();
			
			_objKey.setIndexes(row,col);
			_objValueHelp.setValue(lvalue);
			_objValue.setBaseObject(_objValueHelp);
			
			out.collect(_objKey, _objValue);
		}	
	}
	
	protected static class ResultMergeMapperBinaryCell extends ResultMergeMapper
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
	
	protected static class ResultMergeMapperBinaryBlock extends ResultMergeMapper
	{
		private ResultMergeTaggedMatrixIndexes _objKey;
		private TaggedMatrixBlock _objValue;
		private long _rlen = -1;
		private long _clen = -1;
		private long _brlen = -1;
		private long _bclen = -1;
		
		protected ResultMergeMapperBinaryBlock(byte tag, long rlen, long clen, long brlen, long bclen)
		{
			super(tag);
			_objKey = new ResultMergeTaggedMatrixIndexes();
			_objValue = new TaggedMatrixBlock();
			_objKey.setTag( _tag );
			_objValue.setTag( _tag );
			
			_rlen = rlen;
			_clen = clen;
			_brlen = brlen;
			_bclen = bclen;
		}

		@Override
		protected void processKeyValue(Writable key, Writable value, OutputCollector<Writable, Writable> out, Reporter reporter)
				throws IOException 
		{
			MatrixIndexes inkey = (MatrixIndexes)key;
			MatrixBlock inval = (MatrixBlock)value;
			
			//check valid block sizes
			if( inval.getNumRows() != UtilFunctions.computeBlockSize(_rlen, inkey.getRowIndex(), _brlen) )
				throw new IOException("Invalid number of rows for block "+inkey+": "+inval.getNumRows());
			if( inval.getNumColumns() != UtilFunctions.computeBlockSize(_clen, inkey.getColumnIndex(), _bclen) )
				throw new IOException("Invalid number of columns for block "+inkey+": "+inval.getNumColumns());
			
			//pass-through matrix blocks
			_objKey.getIndexes().setIndexes( inkey );
			_objValue.setBaseObject( inval );
			out.collect(_objKey, _objValue);
		}	
	}
}
