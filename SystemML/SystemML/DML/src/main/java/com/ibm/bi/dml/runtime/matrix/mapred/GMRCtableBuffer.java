/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.util.HashMap;
import java.util.Vector;
import java.util.Map.Entry;

import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;


public class GMRCtableBuffer 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//buffer size is tradeoff between preaggregation and efficient hash probes
	//4k entries * ~64byte = 256KB (common L2 cache size)
	public static final int MAX_BUFFER_SIZE = 4096; 
	
	private HashMap<Byte, HashMap<MatrixIndexes, Double>> _buffer = null;
	private CollectMultipleConvertedOutputs _collector = null;

	private byte[] _resultIndexes = null;
	private long[] _resultNonZeros = null;
	private byte[] _resultDimsUnknown = null;
	private long[] _resultMaxRowDims = null;
	private long[] _resultMaxColDims = null;
	
	
	public GMRCtableBuffer( CollectMultipleConvertedOutputs collector )
	{
		_buffer = new HashMap<Byte, HashMap<MatrixIndexes, Double>>();
		_collector = collector;
	}
	
	/**
	 * 
	 * @param resultIndexes
	 * @param resultsNonZeros
	 * @param resultDimsUnknown
	 * @param resultsMaxRowDims
	 * @param resultsMaxColDims
	 */
	public void setMetadataReferences(byte[] resultIndexes, long[] resultsNonZeros, byte[] resultDimsUnknown, long[] resultsMaxRowDims, long[] resultsMaxColDims)
	{
		_resultIndexes = resultIndexes;
		_resultNonZeros = resultsNonZeros;
		_resultDimsUnknown = resultDimsUnknown;
		_resultMaxRowDims = resultsMaxRowDims;
		_resultMaxColDims = resultsMaxColDims;
	}

	/**
	 * 
	 * @return
	 */
	public int getBufferSize()
	{
		int ret = 0;
		for( Entry<Byte, HashMap<MatrixIndexes, Double>> ctable : _buffer.entrySet() )
			ret += ctable.getValue().size();
		return ret;
	}
	
	/**
	 * 
	 * @return
	 */
	public HashMap<Byte, HashMap<MatrixIndexes, Double>> getBuffer()
	{
		return _buffer;
	}
	
	/**
	 * 
	 * @param reporter
	 * @throws RuntimeException
	 */
	public void flushBuffer( Reporter reporter ) 
		throws RuntimeException 
	{
		try
		{
			MatrixIndexes key=null;//new MatrixIndexes();
			MatrixCell value=new MatrixCell();
			for(Entry<Byte, HashMap<MatrixIndexes, Double>> ctable: _buffer.entrySet())
			{
				Vector<Integer> resultIDs=ReduceBase.getOutputIndexes(ctable.getKey(), _resultIndexes);
				for(Entry<MatrixIndexes, Double> e: ctable.getValue().entrySet())
				{
					key = e.getKey();
					value.setValue(e.getValue());
					for(Integer i: resultIDs)
					{
						_collector.collectOutput(key, value, i, reporter);
						_resultNonZeros[i]++;
						
						if( _resultDimsUnknown[i] == (byte) 1 ) {
							_resultMaxRowDims[i] = Math.max( key.getRowIndex(), _resultMaxRowDims[i]);
							_resultMaxColDims[i] = Math.max( key.getColumnIndex(), _resultMaxColDims[i]);
						}
					}
				}
			}
		}
		catch(Exception ex)
		{
			throw new RuntimeException("Failed to flush ctable buffer.", ex);
		}
		//remove existing partial ctables
		_buffer.clear();
	}
}
