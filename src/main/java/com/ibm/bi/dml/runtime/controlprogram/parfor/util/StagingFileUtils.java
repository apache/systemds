/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.LinkedList;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.util.FastStringTokenizer;

public class StagingFileUtils 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final int CELL_BUFFER_SIZE = 100000;
	
	/**
	 * 
	 * @param fname
	 * @param buffer
	 * @throws IOException
	 */
	public static void writeCellListToLocal( String fname, LinkedList<Cell> buffer ) 
		throws IOException
	{
		FileOutputStream fos = new FileOutputStream( fname, true );
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(fos));	
		try 
		{
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();
			
			for( Cell c : buffer )
			{
				sb.append(c.getRow());
				sb.append(' ');
				sb.append(c.getCol());
				sb.append(' ');
				sb.append(c.getValue());
				sb.append('\n');
				out.write( sb.toString() );
				sb.setLength(0);
			}
		}
		finally
		{
			if( out != null )
				out.close();	
		}	
	}

	public static void writeKeyMappingToLocal( String fname, long[][] keys ) 
		throws IOException
	{
		FileOutputStream fos = new FileOutputStream( fname, true );
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(fos));	
		try 
		{
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();
			
			for( long[] key : keys )
			{
				sb.append(key[0]);
				sb.append(' ');
				sb.append(key[1]);
				sb.append('\n');
				out.write( sb.toString() );
				sb.setLength(0);
			}
		}
		finally
		{
			if( out != null )
				out.close();	
		}	
	}

	public static BufferedReader openKeyMap( String name ) 
		throws FileNotFoundException
	{
		FileInputStream fis = new FileInputStream( name );
		BufferedReader in = new BufferedReader(new InputStreamReader(fis));	
		return in;
	}
	
	public static void nextKeyMap( BufferedReader in, HashMap<Integer,HashMap<Long,Long>> map, int bi, int blen ) 
		throws NumberFormatException, IOException
	{
		String value = null;
		FastStringTokenizer st = new FastStringTokenizer(' ');
		while( (value=in.readLine())!=null )
		{
			st.reset( value ); //reset tokenizer
			long row1 = st.nextLong();
			long row2 = st.nextLong();
			
			int id = (int)row1/blen;
			if( !map.containsKey(id) )
				map.put(id, new HashMap<Long,Long>());
			
			map.get(id).put(row1, row2);
			if( id > bi )
				break;
		}
	}
	
	public static int nextSizedKeyMap( BufferedReader in, HashMap<Integer,HashMap<Long,Long>> map, int blen, int size ) 
		throws NumberFormatException, IOException
	{
		map.clear();
		
		String value = null;
		int len = 0;
		FastStringTokenizer st = new FastStringTokenizer(' ');
		while( (value=in.readLine())!=null )
		{
			st.reset( value ); //reset tokenizer
			long row1 = st.nextLong();
			long row2 = st.nextLong();
			
			int id = (int)row1/blen;
			if( !map.containsKey(id) )
				map.put(id, new HashMap<Long,Long>());
			
			map.get(id).put(row1, row2);
			len++;
			
			if( len >= size )
				break;
		}
		
		return len;
	}
	
	public static void closeKeyMap( BufferedReader in ) 
		throws IOException
	{
		if( in != null )
			in.close();		
	}
	
	/**
	 * 
	 * @param fname
	 * @return
	 * @throws IOException
	 */
	public static LinkedList<Cell> readCellListFromLocal( String fname ) 
		throws IOException
	{
		FileInputStream fis = new FileInputStream( fname );
		BufferedReader in = new BufferedReader(new InputStreamReader(fis));	
		LinkedList<Cell> buffer = new LinkedList<Cell>();
		try 
		{
			String value = null;
			FastStringTokenizer st = new FastStringTokenizer(' '); 
			while( (value=in.readLine())!=null )
			{
				st.reset( value ); //reset tokenizer
				long row = st.nextLong();
				long col = st.nextLong();
				double lvalue = st.nextDouble();
				Cell c =  new Cell( row, col, lvalue );
				buffer.addLast( c );
			}
		}
		finally
		{
			if( in != null )
				in.close();
		}
   		
		return buffer;
	}

	public static MatrixBlock readCellList2BlockFromLocal( String fname, int brlen, int bclen ) 
		throws IOException, DMLRuntimeException 
	{
		return readCellList2BlockFromLocal( fname, brlen, bclen, false );
	}
	
	public static MatrixBlock readCellList2BlockFromLocal( String fname, int brlen, int bclen, boolean sparse ) 
		throws IOException, DMLRuntimeException 
	{
		MatrixBlock tmp = new MatrixBlock( brlen, bclen, sparse );
		if( !sparse )
			tmp.allocateDenseBlockUnsafe(brlen, bclen);
		
		FileInputStream fis = new FileInputStream( fname );
		BufferedReader in = new BufferedReader(new InputStreamReader(fis));	
		FastStringTokenizer st = new FastStringTokenizer(' ');
		try 
		{
			String value = null;
			if( sparse )
			{
				while( (value=in.readLine())!=null )
				{
					st.reset( value ); //reset tokenizer
					int row = st.nextInt();
					int col = st.nextInt();
					double lvalue = st.nextDouble();
					tmp.quickSetValue(row, col, lvalue);
				}
			}
			else
			{
				while( (value=in.readLine())!=null )
				{
					st.reset( value ); //reset tokenizer
					int row = st.nextInt();
					int col = st.nextInt();
					double lvalue = st.nextDouble();
					tmp.setValueDenseUnsafe(row, col, lvalue);
				}
				
				tmp.recomputeNonZeros();
			}
		}
		finally
		{
			if( in != null )
				in.close();
		}
			
		//finally change internal representation if required
		tmp.examSparsity();
		
		return tmp;
	}
	
}
