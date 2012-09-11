package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;

/**
 * Remote data partitioner implementation, realized as MR mapper.
 *
 */
@SuppressWarnings("unchecked")
public class DataPartitionerRemoteMapper 
	implements Mapper<Writable, Writable, Writable, Writable>
{	
	private DataPartitionerMapper _mapper = null;
	
	public DataPartitionerRemoteMapper( ) 
	{
		
	}
	
	/**
	 * 
	 */
	public void map(Writable key, Writable value, OutputCollector<Writable, Writable> out, Reporter reporter) 
		throws IOException
	{
		_mapper.processKeyValue(key, value, reporter);		
	}

	/**
	 * 
	 */
	public void configure(JobConf job)
	{
		long rlen = MRJobConfiguration.getPartitioningNumRows( job );
		long clen = MRJobConfiguration.getPartitioningNumCols( job );
		int brlen = MRJobConfiguration.getPartitioningBlockNumRows( job );
		int bclen = MRJobConfiguration.getPartitioningBlockNumCols( job );
		InputInfo ii = MRJobConfiguration.getPartitioningInputInfo( job );
		PDataPartitionFormat pdf = MRJobConfiguration.getPartitioningFormat( job );
		
		MultipleOutputs mos = new MultipleOutputs(job);
		
		if( ii == InputInfo.TextCellInputInfo )
			_mapper = new DataPartitionerMapperTextcell(mos, rlen, clen, brlen, bclen, pdf);
		else if( ii == InputInfo.BinaryCellInputInfo )
			_mapper = new DataPartitionerMapperBinarycell(mos, rlen, clen, brlen, bclen, pdf);
		else if( ii == InputInfo.BinaryBlockInputInfo )
			_mapper = new DataPartitionerMapperBinaryblock(mos, rlen, clen, brlen, bclen, pdf);
		else
			throw new RuntimeException("Unable to configure mapper with unknown input info: "+ii.toString());
	}
	
	/**
	 * 
	 */
	@Override
	public void close() throws IOException 
	{
		if( _mapper != null )
			_mapper.close();
	}
	
	
	private abstract class DataPartitionerMapper //NOTE: could also be refactored as three different mappers
	{
		protected MultipleOutputs _mos = null;
		
		protected long _rlen = -1;
		protected long _clen = -1;
		protected int _brlen = -1;
		protected int _bclen = -1;
		
		protected PDataPartitionFormat _pdf = null;
	
		
		protected DataPartitionerMapper( MultipleOutputs mos, long rlen, long clen, int brlen, int bclen, PDataPartitionFormat pdf )
		{
			_mos = mos;
			
			_rlen = rlen;
			_clen = clen;
			_brlen = brlen;
			_bclen = bclen;
			_pdf = pdf;
		}
		
		protected abstract void processKeyValue( Writable key, Writable value, Reporter reporter ) 
			throws IOException;
		
		protected void close() 
			throws IOException
		{
			_mos.close();
		}
	}
	
	private class DataPartitionerMapperTextcell extends DataPartitionerMapper
	{
		protected DataPartitionerMapperTextcell(MultipleOutputs mos, long rlen, long clen, int brlen, int bclen, PDataPartitionFormat pdf) 
		{
			super(mos, rlen, clen, brlen, bclen, pdf);
		}

		@Override
		protected void processKeyValue(Writable key, Writable value, Reporter reporter) 
			throws IOException 
		{
			long row = -1;
			long col = -1;
			
			try
			{
				String cellStr = ((Text)value).toString().trim();		
				StringTokenizer st = new StringTokenizer(cellStr, " ");
				row = Long.parseLong( st.nextToken() );
				col = Long.parseLong( st.nextToken() );
				double lvalue = Double.parseDouble( st.nextToken() );
				
				String outName = null;
				switch( _pdf )
				{
					case ROW_WISE:
						outName = String.valueOf(row);
						row = 1;
						break;
					case ROW_BLOCK_WISE:
						outName = String.valueOf( (row-1)/_brlen+1 );
						row = (row-1)%_brlen+1;
						break;
					case COLUMN_WISE:
						outName = String.valueOf(col);
						col = 1;
						break;
					case COLUMN_BLOCK_WISE:
						outName = String.valueOf( (col-1)/_bclen+1 );
						col = (col-1)%_bclen+1;
						break;
				}
				
				StringBuilder sb = new StringBuilder();
				sb.append(row);
				sb.append(" ");
				sb.append(col);
				sb.append(" ");
				sb.append(lvalue);
				Text outValue = new Text(sb.toString());
					
				OutputCollector<NullWritable, Text> out = _mos.getCollector(outName, reporter);
				out.collect(NullWritable.get(), outValue);	
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
		protected DataPartitionerMapperBinarycell(MultipleOutputs mos, long rlen, long clen, int brlen, int bclen, PDataPartitionFormat pdf) 
		{
			super(mos, rlen, clen, brlen, bclen, pdf);
		}

		@Override
		protected void processKeyValue(Writable key, Writable value, Reporter reporter) 
			throws IOException 
		{
			long row = -1;
			long col = -1;

			try
			{
				MatrixIndexes key2 = (MatrixIndexes)key;
				MatrixCell value2 = (MatrixCell)value;
				row = key2.getRowIndex();
				col = key2.getColumnIndex();
				
				String outName = null;
				switch( _pdf )
				{
					case ROW_WISE:
						outName = String.valueOf(row);
						row = 1;
						break;
					case ROW_BLOCK_WISE:
						outName = String.valueOf( (row-1)/_brlen+1 );
						row = (row-1)%_brlen+1;
						break;
					case COLUMN_WISE:
						outName = String.valueOf(col);
						col = 1;
						break;
					case COLUMN_BLOCK_WISE:
						outName = String.valueOf( (col-1)/_bclen+1 );
						col = (col-1)%_bclen+1;
						break;
				}
				key2.setIndexes(row, col);	
				
				OutputCollector<MatrixIndexes, MatrixCell> out = _mos.getCollector(outName, reporter);
				out.collect(key2, value2); 
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
		protected DataPartitionerMapperBinaryblock(MultipleOutputs mos, long rlen, long clen, int brlen, int bclen, PDataPartitionFormat pdf) 
		{
			super(mos, rlen, clen, brlen, bclen, pdf);
		}

		@Override
		protected void processKeyValue(Writable key, Writable value, Reporter reporter) 
			throws IOException 
		{
			try
			{
				MatrixIndexes key2 =  (MatrixIndexes)key;
				MatrixBlock value2 = (MatrixBlock)value;
				long row_offset = (key2.getRowIndex()-1)*_brlen;
				long col_offset = (key2.getColumnIndex()-1)*_bclen;
				
				boolean sparse = value2.isInSparseFormat();
				int rows = value2.getNumRows();
				int cols = value2.getNumColumns();
	
				//bound check per block
				if( row_offset + rows < 1 || row_offset + rows > _rlen || col_offset + cols<1 || col_offset + cols > _clen )
				{
					throw new IOException("Matrix block ["+(row_offset+1)+":"+(row_offset+rows)+","+(col_offset+1)+":"+(col_offset+cols)+"] " +
							              "out of overall matrix range [1:"+_rlen+",1:"+_clen+"].");
				}
				
				String outName = null;
				OutputCollector<MatrixIndexes, MatrixBlock> out = null;
				MatrixBlock tmp = null;
					
				switch( _pdf )
				{
					case ROW_WISE:
						tmp = new MatrixBlock( 1, cols, false ); 
						tmp.spaceAllocForDenseUnsafe(1, cols);				
						for( int i=0; i<rows; i++ )
						{
							outName= String.valueOf( (row_offset+1+i) );
							key2.setIndexes(1, (col_offset/_bclen+1) );						
							if( sparse )
							{
								for( int j=0; j<cols; j++ )
								{
									double lvalue = value2.getValueSparseUnsafe(i, j);
									tmp.setValueDenseUnsafe(0, j, lvalue);
								}
							}
							else
							{
								for( int j=0; j<cols; j++ )
								{
									double lvalue = value2.getValueDenseUnsafe(i, j);
									tmp.setValueDenseUnsafe(0, j, lvalue);
								}
							}
							tmp.recomputeNonZeros();
							out = _mos.getCollector(outName, reporter);
							out.collect(key2, tmp);
						}
						break;
					case ROW_BLOCK_WISE:
						outName= String.valueOf( (row_offset/_brlen+1) );
						key2.setIndexes(1, (col_offset/_bclen+1) );
						out = _mos.getCollector(outName, reporter);
						out.collect(key2, value2);
						break;
					case COLUMN_WISE:
						tmp = new MatrixBlock( rows, 1, false ); 
						tmp.spaceAllocForDenseUnsafe(rows, 1);
						for( int i=0; i<cols; i++ )
						{
							outName= String.valueOf( (col_offset+1+i) );
							key2.setIndexes((row_offset/_brlen+1), 1 );
							if( sparse )
							{
								for( int j=0; j<rows; j++ )
								{
									double lvalue = value2.getValueSparseUnsafe(j, i);
									tmp.setValueDenseUnsafe(j, 0, lvalue);
								}
							}
							else
							{
								for( int j=0; j<rows; j++ )
								{
									double lvalue = value2.getValueDenseUnsafe(j, i);
									tmp.setValueDenseUnsafe(j, 0, lvalue);
								}					
							}
							tmp.recomputeNonZeros();
							out = _mos.getCollector(outName, reporter);
							out.collect(key2, tmp);
						}	
						break;
					case COLUMN_BLOCK_WISE:
						outName= String.valueOf( (col_offset/_bclen+1) );
						key2.setIndexes( (row_offset/_brlen+1), 1 );
						out = _mos.getCollector(outName, reporter);
						out.collect(key2, value2);
						break;
				}
			} 
			catch (Exception e) 
			{
				throw new IOException("Unable to partition binary block matrix.", e);
			}
		}
	}
}
