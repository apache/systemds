/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.IOException;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.TaggedMatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.TaggedMatrixCell;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.DataConverter;

/**
 * Remote result merge reducer that receives all worker results partitioned by
 * cell index or blockindex and merges all results. Due to missing resettable iterators
 * in the old mapred API we need to spill parts of the value list to disk before merging
 * in case of binaryblock.
 *
 */
public class ResultMergeRemoteReducer 
	implements Reducer<Writable, Writable, Writable, Writable>
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private ResultMergeReducer _reducer = null;
	
	public ResultMergeRemoteReducer( ) 
	{
		
	}
	
	@Override
	public void reduce(Writable key, Iterator<Writable> valueList, OutputCollector<Writable, Writable> out, Reporter reporter)
		throws IOException 
	{
		_reducer.processKeyValueList(key, valueList, out, reporter);
	}

	/**
	 * 
	 */
	public void configure(JobConf job)
	{
		InputInfo ii = MRJobConfiguration.getResultMergeInputInfo(job);
		String compareFname = MRJobConfiguration.getResultMergeInfoCompareFilename(job);
		String stagingDir = MRJobConfiguration.getResultMergeStagingDir(job);
		
		//determine compare required
		boolean requiresCompare = false;
		if( !compareFname.equals("null") )
			requiresCompare = true;
		
		if( ii == InputInfo.TextCellInputInfo )
			_reducer = new ResultMergeReducerTextCell(requiresCompare);
		else if( ii == InputInfo.BinaryCellInputInfo )
			_reducer = new ResultMergeReducerBinaryCell(requiresCompare);
		else if( ii == InputInfo.BinaryBlockInputInfo )
			_reducer = new ResultMergeReducerBinaryBlock(requiresCompare, stagingDir);
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

	
	private interface ResultMergeReducer //interface in order to allow ResultMergeReducerBinaryBlock to inherit from ResultMerge
	{	
		void processKeyValueList( Writable key, Iterator<Writable> valueList, OutputCollector<Writable, Writable> out, Reporter reporter ) 
			throws IOException;
	}
	
	private class ResultMergeReducerTextCell implements ResultMergeReducer
	{
		private boolean _requiresCompare;
		private StringBuilder _sb = null;
		private Text _objValue = null;
		
		public ResultMergeReducerTextCell(boolean requiresCompare)
		{
			_requiresCompare = requiresCompare;
			_sb = new StringBuilder();
			_objValue = new Text();
		}
		
		@Override
		public void processKeyValueList(Writable key, Iterator<Writable> valueList, OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException 
		{
			//with compare
			if( _requiresCompare )
			{
				// NOTES MB:
				// 1) the old mapred api does not support multiple scans (reset/mark),
				//    once we switch to the new api, we can use the resetableiterator for doing
				//    the two required scans (finding the compare obj, compare all other objs)
				// 2) for 'textcell' we assume that the entire valueList fits into main memory
				//    this is valid as we group by cells, i.e., we would need millions of input files
				//    to exceed the usual 100-600MB per reduce task.
				
				//scan for compare object (incl result merge if compare available)
				MatrixIndexes key2 = (MatrixIndexes) key;
				Double cellCompare = null;
				Collection<Double> cellList = new LinkedList<Double>();
				boolean found = false;
				while( valueList.hasNext() ) {
					TaggedMatrixCell tVal = (TaggedMatrixCell) valueList.next();
					double lvalue = ((MatrixCell)tVal.getBaseObject()).getValue();
					if( tVal.getTag()==ResultMergeRemoteMR.COMPARE_TAG )
						cellCompare = lvalue;
					else 
					{
						if( cellCompare == null )
							cellList.add( lvalue );
						else if( cellCompare.doubleValue()!=lvalue ) //compare on the fly
						{
							_sb.append(key2.getRowIndex());
							_sb.append(' ');
							_sb.append(key2.getColumnIndex());
							_sb.append(' ');
							_sb.append(lvalue);
							_objValue.set( _sb.toString() );
							_sb.setLength(0);
							out.collect(NullWritable.get(), _objValue );	
							found = true;
							break; //only one write per cell possible (independence)
						}// note: objs with equal value are directly discarded
					}
				}
				
				//result merge for objs before compare
				if( !found )
					for( Double c : cellList )
						if( !c.equals( cellCompare ) )
						{
							_sb.append(key2.getRowIndex());
							_sb.append(' ');
							_sb.append(key2.getColumnIndex());
							_sb.append(' ');
							_sb.append(c.doubleValue());
							_objValue.set( _sb.toString() );
							_sb.setLength(0);							
							out.collect(NullWritable.get(), _objValue );	
							break; //only one write per cell possible (independence)
						}
			}
			//without compare
			else
			{
				MatrixIndexes key2 = (MatrixIndexes) key;
				while( valueList.hasNext() )  
				{
					TaggedMatrixCell tVal = (TaggedMatrixCell) valueList.next(); 
					MatrixCell value = (MatrixCell) tVal.getBaseObject();
					
					_sb.append(key2.getRowIndex());
					_sb.append(' ');
					_sb.append(key2.getColumnIndex());
					_sb.append(' ');
					_sb.append(value.getValue());
					_objValue.set( _sb.toString() );
					_sb.setLength(0);				
					out.collect(NullWritable.get(), _objValue );	
					break; //only one write per cell possible (independence)
				}
			}
			
			
		}
	}
	
	private class ResultMergeReducerBinaryCell implements ResultMergeReducer
	{
		private boolean _requiresCompare;
		private MatrixCell _objValue;
		
		public ResultMergeReducerBinaryCell(boolean requiresCompare)
		{
			_requiresCompare = requiresCompare;
			_objValue = new MatrixCell();
		}

		@Override
		public void processKeyValueList(Writable key, Iterator<Writable> valueList, OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException 
		{
			//with compare
			if( _requiresCompare )
			{
				// NOTES MB:
				// 1) the old mapred api does not support multiple scans (reset/mark),
				//    once we switch to the new api, we can use the resetableiterator for doing
				//    the two required scans (finding the compare obj, compare all other objs)
				// 2) for 'binarycell' we assume that the entire valueList fits into main memory
				//    this is valid as we group by cells, i.e., we would need millions of input files
				//    to exceed the usual 100-600MB per reduce task.
				
				//scan for compare object (incl result merge if compare available)
				Double cellCompare = null;
				Collection<Double> cellList = new LinkedList<Double>();
				boolean found = false;
				while( valueList.hasNext() ) {
					TaggedMatrixCell tVal = (TaggedMatrixCell) valueList.next();
					MatrixCell cVal = (MatrixCell) tVal.getBaseObject();
					if( tVal.getTag()==ResultMergeRemoteMR.COMPARE_TAG )
						cellCompare = cVal.getValue();
					else 
					{
						if( cellCompare == null )
							cellList.add( cVal.getValue() );
						else if( cellCompare.doubleValue() != cVal.getValue() ) //compare on the fly
						{
							out.collect(key, cVal );	
							found = true;
							break; //only one write per cell possible (independence)
						}// note: objs with equal value are directly discarded
					}
				}
				
				//result merge for objs before compare
				if( !found )
					for( Double c : cellList )				
						if( !c.equals( cellCompare) )
						{				
							_objValue.setValue(c.doubleValue());
							out.collect(key, _objValue );	
							break; //only one write per cell possible (independence)
						}
			}
			//without compare
			else
			{
				while( valueList.hasNext() )  
				{
					TaggedMatrixCell tVal = (TaggedMatrixCell) valueList.next(); 
					out.collect((MatrixIndexes)key, (MatrixCell)tVal.getBaseObject());	
					break; //only one write per cell possible (independence)
				}
			}
		}
	}
	
	private class ResultMergeReducerBinaryBlock extends ResultMerge implements ResultMergeReducer
	{
		private boolean _requiresCompare;
		private String _stagingDir;
		
		public ResultMergeReducerBinaryBlock(boolean requiresCompare, String stagingDir)
		{
			_requiresCompare = requiresCompare;
			_stagingDir = stagingDir;
		}
		
		@Override
		public MatrixObject executeParallelMerge(int par) 
			throws DMLRuntimeException 
		{
			throw new DMLRuntimeException("Unsupported operation.");
		}

		@Override
		public MatrixObject executeSerialMerge() 
			throws DMLRuntimeException 
		{
			throw new DMLRuntimeException("Unsupported operation.");
		}

		@Override
		public void processKeyValueList(Writable key, Iterator<Writable> valueList, OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException 
		{	
			try
			{
				MatrixIndexes ixOut = ((ResultMergeTaggedMatrixIndexes)key).getIndexes();
				MatrixBlock mbOut = null;
				double[][] aCompare = null;
				boolean appendOnly = false;
				
				//get and prepare compare block if required
				if( _requiresCompare )
				{
					TaggedMatrixBlock tVal = (TaggedMatrixBlock) valueList.next();
					MatrixBlock bVal = (MatrixBlock) tVal.getBaseObject();
					if( tVal.getTag()!=ResultMergeRemoteMR.COMPARE_TAG )
						throw new IOException("Failed to read compare block at expected first position.");
					aCompare = DataConverter.convertToDoubleMatrix(bVal);
				}
				
				//merge all result blocks into final result block 
				while( valueList.hasNext() ) 
				{
					TaggedMatrixBlock tVal = (TaggedMatrixBlock) valueList.next();
					MatrixBlock bVal = (MatrixBlock) tVal.getBaseObject();
					
					if( mbOut == null ) //copy first block
					{
						mbOut = new MatrixBlock();
						mbOut.copy( bVal );
						appendOnly = mbOut.isInSparseFormat();
					}
					else //merge remaining blocks
					{
						if( _requiresCompare )
							mergeWithComp(mbOut, bVal, aCompare);
						else
							mergeWithoutComp(mbOut, bVal, appendOnly);	
					}
				}
				
				//sort sparse due to append-only
				if( appendOnly )
					mbOut.sortSparseRows();
				
				//change sparsity if required after 
				mbOut.examSparsity(); 
				
				out.collect(ixOut, mbOut);
			}
			catch( Exception ex )
			{
				throw new IOException(ex);
			}			
		}
		
	}
}
