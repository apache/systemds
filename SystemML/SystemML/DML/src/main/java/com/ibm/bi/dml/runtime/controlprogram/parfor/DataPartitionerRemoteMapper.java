/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.PairWritableBlock;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.PairWritableCell;
import com.ibm.bi.dml.runtime.matrix.io.IJV;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.SparseRowsIterator;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.FastStringTokenizer;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

/**
 * Remote data partitioner mapper implementation that does the actual
 * partitioning and key creation according to the given format.
 *
 */
public class DataPartitionerRemoteMapper 
	implements Mapper<Writable, Writable, Writable, Writable>
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private DataPartitionerMapper _mapper = null;
	
	public DataPartitionerRemoteMapper( ) 
	{
		
	}
	
	/**
	 * 
	 */
	@Override
	public void map(Writable key, Writable value, OutputCollector<Writable, Writable> out, Reporter reporter) 
		throws IOException
	{
		_mapper.processKeyValue(key, value, out, reporter);		
	}

	/**
	 * 
	 */
	@Override
	public void configure(JobConf job)
	{
		long rlen = MRJobConfiguration.getPartitioningNumRows( job );
		long clen = MRJobConfiguration.getPartitioningNumCols( job );
		int brlen = MRJobConfiguration.getPartitioningBlockNumRows( job );
		int bclen = MRJobConfiguration.getPartitioningBlockNumCols( job );
		InputInfo ii = MRJobConfiguration.getPartitioningInputInfo( job );
		OutputInfo oi = MRJobConfiguration.getPartitioningOutputInfo( job );
		PDataPartitionFormat pdf = MRJobConfiguration.getPartitioningFormat( job );
		int n = MRJobConfiguration.getPartitioningSizeN( job );
		boolean keepIndexes =  MRJobConfiguration.getPartitioningIndexFlag( job );
		
		if( ii == InputInfo.TextCellInputInfo )
			_mapper = new DataPartitionerMapperTextcell(rlen, clen, brlen, bclen, pdf, n);
		else if( ii == InputInfo.BinaryCellInputInfo )
			_mapper = new DataPartitionerMapperBinarycell(rlen, clen, brlen, bclen, pdf, n);
		else if( ii == InputInfo.BinaryBlockInputInfo )
		{
			if( oi == OutputInfo.BinaryBlockOutputInfo )
				_mapper = new DataPartitionerMapperBinaryblock(rlen, clen, brlen, bclen, pdf, n, keepIndexes);
			else if( oi == OutputInfo.BinaryCellOutputInfo )
			{
				boolean outputEmpty = MRJobConfiguration.getProgramBlocks(job)!=null; //fused parfor
				_mapper = new DataPartitionerMapperBinaryblock2Binarycell(job, rlen, clen, brlen, bclen, pdf, n, keepIndexes, outputEmpty); 
			}
			else
				throw new RuntimeException("Partitioning from '"+ii+"' to '"+oi+"' not supported");
		}
		else
			throw new RuntimeException("Unable to configure mapper with unknown input info: "+ii.toString());
	}
	
	/**
	 * 
	 */
	@Override
	public void close() 
		throws IOException 
	{
		_mapper.close();
	}
	
	private abstract class DataPartitionerMapper //NOTE: could also be refactored as three different mappers
	{
		protected long _rlen = -1;
		protected long _clen = -1;
		protected int _brlen = -1;
		protected int _bclen = -1;
		protected PDataPartitionFormat _pdf = null;
		protected int _n = -1;
	
		protected DataPartitionerMapper( long rlen, long clen, int brlen, int bclen, PDataPartitionFormat pdf, int n )
		{
			_rlen = rlen;
			_clen = clen;
			_brlen = brlen;
			_bclen = bclen;
			_pdf = pdf;
			_n = n;
		}
		
		protected abstract void processKeyValue( Writable key, Writable value, OutputCollector<Writable, Writable> out, Reporter reporter ) 
			throws IOException;
		
		protected void close()
			throws IOException
		{
			//do nothing
		}
	}
	
	private class DataPartitionerMapperTextcell extends DataPartitionerMapper
	{
		private FastStringTokenizer _st = null;
		
		protected DataPartitionerMapperTextcell(long rlen, long clen, int brlen, int bclen, PDataPartitionFormat pdf, int n) 
		{
			super(rlen, clen, brlen, bclen, pdf, n);
			_st = new FastStringTokenizer(' ');
		}

		@Override
		protected void processKeyValue(Writable key, Writable value, OutputCollector<Writable, Writable> out, Reporter reporter) 
			throws IOException 
		{
			long row = -1;
			long col = -1;
			
			try
			{
				_st.reset( value.toString() ); //reset tokenizer
				row = _st.nextLong();
				col = _st.nextLong();
				double lvalue = _st.nextDouble();
				
				LongWritable longKey = new LongWritable();
				PairWritableCell pairValue = new PairWritableCell();
				MatrixIndexes key2 = new MatrixIndexes();
				MatrixCell value2 = new MatrixCell();
				
				switch( _pdf )
				{
					case ROW_WISE:
						longKey.set( row );
						row = 1;
						break;
					case ROW_BLOCK_WISE:
						longKey.set( (row-1)/_brlen+1 );
						row = (row-1)%_brlen+1;
						break;
					case ROW_BLOCK_WISE_N:
						longKey.set( (row-1)/_n+1 );
						row = (row-1)%_n+1;
						break;	
					case COLUMN_WISE:
						longKey.set( col );
						col = 1;
						break;
					case COLUMN_BLOCK_WISE:
						longKey.set( (col-1)/_bclen+1 );
						col = (col-1)%_bclen+1;
						break;
					case COLUMN_BLOCK_WISE_N:
						longKey.set( (col-1)/_n+1 );
						col = (col-1)%_n+1;
						break;	
				}

				key2.setIndexes(row, col);	
				value2.setValue( lvalue );
				pairValue.indexes = key2;
				pairValue.cell = value2;
				out.collect(longKey, pairValue);
			} 
			catch (Exception e) 
			{
				//post-mortem error handling and bounds checking
				if( row < 1 || row > _rlen || col < 1 || col > _clen )
				{
					throw new IOException("Matrix cell ["+(row)+","+(col)+"] " +
										  "out of overall matrix range [1:"+_rlen+",1:"+_clen+"].");
				}
				else
					throw new IOException("Unable to partition text cell matrix.", e);
			}
		}	
	}
	
	private class DataPartitionerMapperBinarycell extends DataPartitionerMapper
	{
		protected DataPartitionerMapperBinarycell(long rlen, long clen, int brlen, int bclen, PDataPartitionFormat pdf, int n) 
		{
			super(rlen, clen, brlen, bclen, pdf, n);
		}

		@Override
		protected void processKeyValue(Writable key, Writable value, OutputCollector<Writable, Writable> out, Reporter reporter) 
			throws IOException 
		{
			long row = -1;
			long col = -1;

			try
			{
				LongWritable longKey = new LongWritable();
				PairWritableCell pairValue = new PairWritableCell();
				MatrixIndexes key2 = (MatrixIndexes)key;
				MatrixCell value2 = (MatrixCell)value;
				row = key2.getRowIndex();
				col = key2.getColumnIndex();
				
				switch( _pdf )
				{
					case ROW_WISE:
						longKey.set(row);
						row = 1;
						break;
					case ROW_BLOCK_WISE:
						longKey.set( (row-1)/_brlen+1 );
						row = (row-1)%_brlen+1; 
						break;
					case ROW_BLOCK_WISE_N:
						longKey.set( (row-1)/_n+1 );
						row = (row-1)%_n+1; 
						break;	
					case COLUMN_WISE:
						longKey.set(col);
						col = 1;
						break;
					case COLUMN_BLOCK_WISE:
						longKey.set( (col-1)/_bclen+1 );
						col = (col-1)%_bclen+1; 
						break;
					case COLUMN_BLOCK_WISE_N:
						longKey.set( (col-1)/_n+1 );
						col = (col-1)%_n+1; 
						break;	
				}
				key2.setIndexes(row, col);	
				pairValue.indexes = key2;
				pairValue.cell = value2;
				out.collect(longKey, pairValue);
			} 
			catch (Exception e) 
			{
				//post-mortem error handling and bounds checking
				if( row < 1 || row > _rlen || col < 1 || col > _clen )
				{
					throw new IOException("Matrix cell ["+(row)+","+(col)+"] " +
										  "out of overall matrix range [1:"+_rlen+",1:"+_clen+"].");
				}
				else
					throw new IOException("Unable to partition binary cell matrix.", e);
			}
		}
	}
	
	private class DataPartitionerMapperBinaryblock extends DataPartitionerMapper
	{
		private MatrixBlock _reuseBlk = null;
		private LongWritable _longKey = null;
		private MatrixIndexes _pairKey = null;
		private PairWritableBlock _pair = null;
		private boolean _keepIndexes = false;
		
		protected DataPartitionerMapperBinaryblock(long rlen, long clen, int brlen, int bclen, PDataPartitionFormat pdf, int n, boolean keepIndexes) 
		{
			super(rlen, clen, brlen, bclen, pdf, n);
		
			//create reuse keys and block 
			_longKey = new LongWritable(); //MR key
			_pair = new PairWritableBlock(); //MR value (pair composed of key and value)
			_pairKey = new MatrixIndexes();
			_reuseBlk = DataPartitioner.createReuseMatrixBlock(pdf, brlen, bclen);
			
			//prewire pair outputs
			_pair.indexes = _pairKey;
			_pair.block = _reuseBlk;
			
			_keepIndexes = keepIndexes;
		}
		
		@Override
		protected void processKeyValue(Writable key, Writable value, OutputCollector<Writable, Writable> out, Reporter reporter) 
			throws IOException 
		{
			try
			{
				MatrixIndexes key2 =  (MatrixIndexes)key;
				MatrixBlock value2 = (MatrixBlock)value;
				long row_offset = (key2.getRowIndex()-1)*_brlen;
				long col_offset = (key2.getColumnIndex()-1)*_bclen;
				
				boolean sparse = value2.isInSparseFormat();
				int nnz = value2.getNonZeros();
				long rows = value2.getNumRows();
				long cols = value2.getNumColumns();
				double sparsity = ((double)nnz)/(rows*cols);
				
				//bound check per block
				if( row_offset + rows < 1 || row_offset + rows > _rlen || col_offset + cols<1 || col_offset + cols > _clen )
				{
					throw new IOException("Matrix block ["+(row_offset+1)+":"+(row_offset+rows)+","+(col_offset+1)+":"+(col_offset+cols)+"] " +
							              "out of overall matrix range [1:"+_rlen+",1:"+_clen+"].");
				}
						
				//partition inputs according to partitioning scheme 
				//(note: output pair is pre-wired, changes only required for blockwise)
				switch( _pdf )
				{
					case ROW_WISE:
						_reuseBlk.reset(1, (int)cols, sparse, (int)(cols*sparsity));								
						for( int i=0; i<rows; i++ )
						{
							_longKey.set(row_offset+1+i);
							_pairKey.setIndexes(1, (col_offset/_bclen+1) );	
							value2.sliceOperations(i+1, i+1, 1, cols, _reuseBlk);
							out.collect(_longKey, _pair);
							_reuseBlk.reset();
						}
						break;
					case ROW_BLOCK_WISE:
						_longKey.set((row_offset/_brlen+1));
						_pairKey.setIndexes(1, (col_offset/_bclen+1) );
						_pair.block = value2;
						out.collect(_longKey, _pair);
						break;
					case ROW_BLOCK_WISE_N:
						_longKey.set((row_offset/_n+1));
						if( _keepIndexes )
							_pairKey.setIndexes(row_offset/_brlen+1, col_offset/_bclen+1 );
						else
							_pairKey.setIndexes(((row_offset%_n)/_brlen)+1, (col_offset/_bclen+1) );
						_pair.block = value2;
						out.collect(_longKey, _pair);
						break;
					case COLUMN_WISE:
						_reuseBlk.reset((int)rows, 1, false);
						for( int i=0; i<cols; i++ )
						{
							_longKey.set(col_offset+1+i);
							_pairKey.setIndexes(row_offset/_brlen+1, 1);							
							value2.sliceOperations(1, rows, i+1, i+1, _reuseBlk);
							out.collect(_longKey, _pair );
							_reuseBlk.reset();
						}	
						break;
					case COLUMN_BLOCK_WISE:
						_longKey.set(col_offset/_bclen+1);
						_pairKey.setIndexes( row_offset/_brlen+1, 1 );
						_pair.block = value2;
						out.collect(_longKey, _pair);
						break;
					case COLUMN_BLOCK_WISE_N:
						_longKey.set(col_offset/_n+1);
						if( _keepIndexes )
							_pairKey.setIndexes( row_offset/_brlen+1, col_offset/_bclen+1 );
						else
							_pairKey.setIndexes( row_offset/_brlen+1, ((col_offset%_n)/_bclen)+1 );
						_pair.block = value2;
						out.collect(_longKey, _pair);
						break;	
				}
			} 
			catch (Exception e) 
			{
				throw new IOException("Unable to partition binary block matrix.", e);
			}
		}	
	}
	
	/**
	 * 
	 */
	private class DataPartitionerMapperBinaryblock2Binarycell extends DataPartitionerMapper
	{
		private JobConf _cachedJobConf = null;
		private boolean _outputEmpty = false;
		
		private boolean _keepIndexes = false;
		
		private OutputCollector<Writable, Writable> _out = null;
		
		protected DataPartitionerMapperBinaryblock2Binarycell(JobConf job, long rlen, long clen, int brlen, int bclen, PDataPartitionFormat pdf, int n, boolean keepIndexes, boolean outputEmpty) 
		{
			super(rlen, clen, brlen, bclen, pdf, n);
			_outputEmpty = outputEmpty;
			_cachedJobConf = job;
			
			_keepIndexes = keepIndexes;
		}
		
		@Override
		protected void processKeyValue(Writable key, Writable value, OutputCollector<Writable, Writable> out, Reporter reporter) 
			throws IOException 
		{
			_out = out;
			
			try
			{
				LongWritable longKey = new LongWritable();
				PairWritableCell pairValue = new PairWritableCell();
				MatrixIndexes key2 =  (MatrixIndexes)key;
				MatrixBlock value2 = (MatrixBlock)value;
				MatrixIndexes cellkey2 = new MatrixIndexes();
				MatrixCell cellvalue2 = new MatrixCell();
				
				long row_offset = (key2.getRowIndex()-1)*_brlen;
				long col_offset = (key2.getColumnIndex()-1)*_bclen;
				
				boolean sparse = value2.isInSparseFormat();
				long rows = value2.getNumRows();
				long cols = value2.getNumColumns();
				
				//bound check per block
				if( row_offset + rows < 1 || row_offset + rows > _rlen || col_offset + cols<1 || col_offset + cols > _clen )
				{
					throw new IOException("Matrix block ["+(row_offset+1)+":"+(row_offset+rows)+","+(col_offset+1)+":"+(col_offset+cols)+"] " +
							              "out of overall matrix range [1:"+_rlen+",1:"+_clen+"].");
				}
							
				long rowBlockIndex = -1;
				long colBlockIndex = -1;
				switch( _pdf )
				{
					case ROW_WISE:
						if( sparse )
						{
							SparseRowsIterator iter = value2.getSparseRowsIterator();
							while( iter.hasNext() )
							{
								IJV lcell = iter.next();
								longKey.set( row_offset + lcell.i + 1 );
								cellkey2.setIndexes( 1, col_offset + lcell.j + 1 );	
								cellvalue2.setValue( lcell.v );
								pairValue.indexes = cellkey2;
								pairValue.cell = cellvalue2;
								out.collect(longKey, pairValue);
							}
						}
						else
							for( int i=0; i<rows; i++ )
							{
								longKey.set(row_offset + i + 1);
								for( int j=0; j<cols; j++ )
								{
									double lvalue = value2.getValueDenseUnsafe(i, j);
									if( lvalue != 0 ) //only for nnz
									{
										cellkey2.setIndexes( 1, col_offset + j + 1 );	
										cellvalue2.setValue( lvalue );
										pairValue.indexes = cellkey2;
										pairValue.cell = cellvalue2;
										out.collect(longKey, pairValue);
									}	
								}
							}	
						break;
					case ROW_BLOCK_WISE:
						longKey.set((row_offset/_brlen+1));
						if( sparse )
						{
							SparseRowsIterator iter = value2.getSparseRowsIterator();
							while( iter.hasNext() )
							{
								IJV lcell = iter.next();
								cellkey2.setIndexes( 1, col_offset + lcell.j + 1 );	
								cellvalue2.setValue( lcell.v );
								pairValue.indexes = cellkey2;
								pairValue.cell = cellvalue2;
								out.collect(longKey, pairValue);
							}
						}
						else
							for( int i=0; i<rows; i++ )
								for( int j=0; j<cols; j++ )
								{
									double lvalue = value2.getValueDenseUnsafe(i, j);
									if( lvalue != 0 ) //only for nnz
									{
										cellkey2.setIndexes( 1, col_offset + j + 1 );	
										cellvalue2.setValue( lvalue );
										pairValue.indexes = cellkey2;
										pairValue.cell = cellvalue2;
										out.collect(longKey, pairValue);
									}	
								}	
						break;
					case ROW_BLOCK_WISE_N:
						longKey.set((row_offset/_n+1));
						if( _keepIndexes )
							rowBlockIndex = ((row_offset)/_brlen)+1;
						else
							rowBlockIndex = ((row_offset%_n)/_brlen)+1;
						if( sparse )
						{
							SparseRowsIterator iter = value2.getSparseRowsIterator();
							while( iter.hasNext() )
							{
								IJV lcell = iter.next();
								cellkey2.setIndexes( rowBlockIndex, col_offset + lcell.j + 1 );	
								cellvalue2.setValue( lcell.v );
								pairValue.indexes = cellkey2;
								pairValue.cell = cellvalue2;
								out.collect(longKey, pairValue);
							}
						}
						else
							for( int i=0; i<rows; i++ )
								for( int j=0; j<cols; j++ )
								{
									double lvalue = value2.getValueDenseUnsafe(i, j);
									if( lvalue != 0 ) //only for nnz
									{
										cellkey2.setIndexes( rowBlockIndex, col_offset + j + 1 );	
										cellvalue2.setValue( lvalue );
										pairValue.indexes = cellkey2;
										pairValue.cell = cellvalue2;
										out.collect(longKey, pairValue);
									}	
								}	
						break;
					case COLUMN_WISE:
						if( sparse )
						{
							SparseRowsIterator iter = value2.getSparseRowsIterator();
							while( iter.hasNext() )
							{
								IJV lcell = iter.next();
								longKey.set( col_offset + lcell.j + 1 );
								cellkey2.setIndexes( row_offset + lcell.i + 1, 1 );	
								cellvalue2.setValue( lcell.v );
								pairValue.indexes = cellkey2;
								pairValue.cell = cellvalue2;
								out.collect(longKey, pairValue);
							}
						}
						else
							for( int j=0; j<cols; j++ )
							{
								longKey.set(col_offset + j + 1);
								for( int i=0; i<rows; i++ )
								{
									double lvalue = value2.getValueDenseUnsafe(i, j);
									if( lvalue != 0 ) //only for nnz
									{
										cellkey2.setIndexes( row_offset + i + 1, 1 );	
										cellvalue2.setValue( lvalue );
										pairValue.indexes = cellkey2;
										pairValue.cell = cellvalue2;
										out.collect(longKey, pairValue);
									}	
								}
							}	
						break;
					case COLUMN_BLOCK_WISE:
						longKey.set(col_offset/_bclen+1);
						if( sparse )
						{
							SparseRowsIterator iter = value2.getSparseRowsIterator();
							while( iter.hasNext() )
							{
								IJV lcell = iter.next();
								cellkey2.setIndexes( row_offset + lcell.i + 1, 1 );	
								cellvalue2.setValue( lcell.v );
								pairValue.indexes = cellkey2;
								pairValue.cell = cellvalue2;
								out.collect(longKey, pairValue);
							}
						}
						else
							for( int j=0; j<cols; j++ )
								for( int i=0; i<rows; i++ )
								{
									double lvalue = value2.getValueDenseUnsafe(i, j);
									if( lvalue != 0 ) //only for nnz
									{
										cellkey2.setIndexes( row_offset + i + 1, 1 );	
										cellvalue2.setValue( lvalue );
										pairValue.indexes = cellkey2;
										pairValue.cell = cellvalue2;
										out.collect(longKey, pairValue);
									}	
								}
						break;
					case COLUMN_BLOCK_WISE_N:
						longKey.set(col_offset/_n+1);
						if( _keepIndexes )
							colBlockIndex = ((col_offset)/_bclen)+1;
						else	
							colBlockIndex = ((col_offset%_n)/_bclen)+1;
						if( sparse )
						{
							SparseRowsIterator iter = value2.getSparseRowsIterator();
							while( iter.hasNext() )
							{
								IJV lcell = iter.next();
								cellkey2.setIndexes( row_offset + lcell.i + 1, colBlockIndex );	
								cellvalue2.setValue( lcell.v );
								pairValue.indexes = cellkey2;
								pairValue.cell = cellvalue2;
								out.collect(longKey, pairValue);
							}
						}
						else
							for( int j=0; j<cols; j++ )
								for( int i=0; i<rows; i++ )
								{
									double lvalue = value2.getValueDenseUnsafe(i, j);
									if( lvalue != 0 ) //only for nnz
									{
										cellkey2.setIndexes( row_offset + i + 1, colBlockIndex );	
										cellvalue2.setValue( lvalue );
										pairValue.indexes = cellkey2;
										pairValue.cell = cellvalue2;
										out.collect(longKey, pairValue);
									}	
								}
						break;
				}
			} 
			catch (Exception e) 
			{
				throw new IOException("Unable to partition binary block matrix.", e);
			}
		}	
		
		@Override
		protected void close() 
			throws IOException
		{
			if( _outputEmpty )
			{
				LongWritable longKey = new LongWritable();
				PairWritableCell pairValue = new PairWritableCell();
				pairValue.indexes = new MatrixIndexes(-1,-1);
				pairValue.cell = new MatrixCell(0);
				
				long mapID = Long.parseLong(MapReduceTool.getUniqueKeyPerTask(_cachedJobConf, true));
				long numMap = _cachedJobConf.getNumMapTasks(); 
				
				//output part of empty blocks (all mappers contribute for better load balance),
				//where mapper responsibility is distributed over all partitions
				long numPartitions = -1;
				switch( _pdf ){
					case ROW_WISE: 			 numPartitions = _rlen; break;
					case ROW_BLOCK_WISE:     numPartitions = (int)Math.ceil(_rlen/(double)_brlen); break;
					case COLUMN_WISE:        numPartitions = _clen; break;
					case COLUMN_BLOCK_WISE:  numPartitions = (int)Math.ceil(_clen/(double)_bclen); break;
				}
				
				long len = (long)Math.ceil((double)numPartitions/numMap);
				long start = mapID * len;
				long end = Math.min((mapID+1) * len, numPartitions);
				for( long i=start; i<end; i++ )
				{
					longKey.set( i+1 );
					_out.collect(longKey, pairValue);
				}	
			}
			
			
		}
	}
}
