package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.OutputCollector;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.matrix.io.AdaptivePartialBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM.IJV;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.PartialBlock;
import com.ibm.bi.dml.runtime.matrix.io.TaggedAdaptivePartialBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM.SparseCellIterator;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class ReblockBuffer 
{
	//default buffer size: 4M -> 96MB
	public static final int DEFAULT_BUFFER_SIZE = 4000000;
	
	private long[]   _buffRows = null;
	private long[]   _buffCols = null;
	private double[] _buffVals = null;
	
	private int _bufflen = -1;
	private int _count = -1;
	
	private long _rlen = -1;
	private long _clen = -1;
	private int _brlen = -1;
	private int _bclen = -1;
	
	public ReblockBuffer( long rlen, long clen, int brlen, int bclen )
	{
		this( DEFAULT_BUFFER_SIZE, rlen, clen, brlen, bclen );
	}
	
	public ReblockBuffer( int buffersize, long rlen, long clen, int brlen, int bclen  )
	{
		//System.out.println("Creating reblock buffer of size "+buffersize);
		
		_bufflen = buffersize;
		_count = 0;
		
		_buffRows = new long[ _bufflen ];
		_buffCols = new long[ _bufflen ];
		_buffVals = new double[ _bufflen ];
		
		_rlen = rlen;
		_clen = clen;
		_brlen = brlen;
		_bclen = bclen;
	}
	
	/**
	 * 
	 * @param r
	 * @param c
	 * @param v
	 */
	public void appendCell( long r, long c, double v )
	{
		_buffRows[ _count ] = r;
		_buffCols[ _count ] = c;
		_buffVals[ _count ] = v;
		_count++;
	}
	
	/**
	 * 
	 * @param r_offset
	 * @param c_offset
	 * @param inBlk
	 * @param index
	 * @param out
	 * @throws IOException
	 */
	public void appendBlock(long r_offset, long c_offset, MatrixBlock inBlk, byte index, OutputCollector<Writable, Writable> out ) 
		throws IOException
	{
		if( inBlk.isInSparseFormat() ) //SPARSE
		{
			SparseCellIterator iter = inBlk.getSparseCellIterator();
			while( iter.hasNext() )
			{
				IJV cell = iter.next();
				_buffRows[ _count ] = r_offset + cell.i;
				_buffCols[ _count ] = c_offset + cell.j;
				_buffVals[ _count ] = cell.v;
				_count++;
				
				//check and flush if required
				if( _count ==_bufflen )
					flushBuffer(index, out);
			}
		}
		else //DENSE
		{
			//System.out.println("dense merge with ro="+r_offset+", co="+c_offset);
			int rlen = inBlk.getNumRows();
			int clen = inBlk.getNumColumns();
			for( int i=0; i<rlen; i++ )
				for( int j=0; j<clen; j++ )
				{
					double val = inBlk.getValueDenseUnsafe(i, j);
					if( val !=0 )
					{
						_buffRows[ _count ] = r_offset + i;
						_buffCols[ _count ] = c_offset + j;
						_buffVals[ _count ] = val;
						_count++;
						
						//check and flush if required
						if( _count ==_bufflen )
							flushBuffer(index, out);
					}
				}
		}
	}
	
	public int getSize()
	{
		return _count;
	}
	
	public int getCapacity()
	{
		return _bufflen;
	}
	
	/**
	 * 
	 * @param index
	 * @param out
	 * @throws IOException
	 */
	public void flushBuffer( byte index, OutputCollector<Writable, Writable> out ) 
		throws IOException
	{
		if( _count == 0 )
			return;
		
		//Timing time = new Timing();
		//time.start();
		
		//Step 1) scan for number of created blocks
		HashSet<MatrixIndexes> IX = new HashSet<MatrixIndexes>();
		MatrixIndexes tmpIx = new MatrixIndexes();
		for( int i=0; i<_count; i++ )
		{
			long bi=UtilFunctions.blockIndexCalculation(_buffRows[i], _brlen);
			long bj=UtilFunctions.blockIndexCalculation(_buffCols[i], _bclen);
			
			tmpIx.setIndexes(bi, bj);
			if( !IX.contains(tmpIx) ){ //probe
				IX.add(tmpIx);
				tmpIx = new MatrixIndexes();
			}
		}
		
		//Step 2) decide on intermediate representation
		boolean blocked = true;
		long blockedSize = IX.size()*_brlen*4 + 12*_count; //worstcase
		long cellSize = 24 * _count;
		if( IX.size()>16 && blockedSize>cellSize )
			blocked = false;
		//System.out.println("Reblock Mapper: output in blocked="+blocked+", num="+IX.size());
		
		//Step 3)
		TaggedAdaptivePartialBlock outTVal = new TaggedAdaptivePartialBlock();
		AdaptivePartialBlock outVal = new AdaptivePartialBlock();
		outTVal.setTag(index);
		outTVal.setBaseObject(outVal); //setup wrapper writables
		if( blocked ) //output binaryblock
		{
			//create intermediate blocks
			boolean sparse = ( OptimizerUtils.getSparsity(_brlen,_bclen,_count/IX.size()) < MatrixBlockDSM.SPARCITY_TURN_POINT 
					            && _clen > MatrixBlockDSM.SKINNY_MATRIX_TURN_POINT );					      
			//System.out.println("blocked with sparse="+sparse);
			HashMap<MatrixIndexes,MatrixBlock> blocks = new HashMap<MatrixIndexes,MatrixBlock>();
			for( MatrixIndexes ix : IX )
			{
				blocks.put(ix, new MatrixBlock(
						Math.min(_brlen, (int)(_rlen-(ix.getRowIndex()-1)*_brlen)),
						Math.min(_bclen, (int)(_clen-(ix.getColumnIndex()-1)*_bclen)),
						sparse));
			}
			
			//put values into blocks
			for( int i=0; i<_count; i++ )
			{
				long bi=UtilFunctions.blockIndexCalculation(_buffRows[i], _brlen);
				long bj=UtilFunctions.blockIndexCalculation(_buffCols[i], _bclen);
				int ci=UtilFunctions.cellInBlockCalculation(_buffRows[i], _brlen);
				int cj=UtilFunctions.cellInBlockCalculation(_buffCols[i], _bclen);
				tmpIx.setIndexes(bi, bj);
				MatrixBlock blk = blocks.get(tmpIx);
				blk.appendValue(ci, cj, _buffVals[i]); //sort on output
			}
			
			//output blocks
			for( Entry<MatrixIndexes,MatrixBlock> e : blocks.entrySet() )
			{
				MatrixIndexes ix = e.getKey();
				MatrixBlock blk = e.getValue();
				blk.sortSparseRows();
				outVal.set(blk); //in outTVal;
				out.collect(ix, outTVal);
			}
		}
		else //output binarycell
		{
			PartialBlock tmpVal = new PartialBlock();
			outVal.set(tmpVal);
			for( int i=0; i<_count; i++ )
			{
				long bi=UtilFunctions.blockIndexCalculation(_buffRows[i], _brlen);
				long bj=UtilFunctions.blockIndexCalculation(_buffCols[i], _bclen);
				int ci=UtilFunctions.cellInBlockCalculation(_buffRows[i], _brlen);
				int cj=UtilFunctions.cellInBlockCalculation(_buffCols[i], _bclen);
				tmpIx.setIndexes(bi, bj);
				tmpVal.set(ci, cj, _buffVals[i]); //in outVal, in outTVal
				out.collect(tmpIx, outTVal);
			}
		}
		
		//System.out.println("flushed buffer (count="+_count+") in "+time.stop());
		
		_count = 0;
	}
}
