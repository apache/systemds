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
import java.util.StringTokenizer;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.utils.DMLRuntimeException;

public class StagingFileUtils 
{
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
		while( (value=in.readLine())!=null )
		{
			String cellStr = value.toString().trim();							
			StringTokenizer st = new StringTokenizer(cellStr, " ");
			long row1 = Long.parseLong( st.nextToken() );
			long row2 = Long.parseLong( st.nextToken() );
			
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
		String value = null;
		int len = 0;
		while( (value=in.readLine())!=null )
		{
			String cellStr = value.toString().trim();							
			StringTokenizer st = new StringTokenizer(cellStr, " ");
			long row1 = Long.parseLong( st.nextToken() );
			long row2 = Long.parseLong( st.nextToken() );
			
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
			long row, col;
			while( (value=in.readLine())!=null )
			{
				String cellStr = value.toString().trim();							
				StringTokenizer st = new StringTokenizer(cellStr, " ");
				row = Long.parseLong( st.nextToken() );
				col = Long.parseLong( st.nextToken() );
				double lvalue = Double.parseDouble( st.nextToken() );
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
			tmp.spaceAllocForDenseUnsafe(brlen, bclen);
		
		FileInputStream fis = new FileInputStream( fname );
		BufferedReader in = new BufferedReader(new InputStreamReader(fis));	
		try 
		{
			String value = null;
			long row, col;
			if( sparse )
			{
				while( (value=in.readLine())!=null )
				{
					String cellStr = value.toString().trim();							
					StringTokenizer st = new StringTokenizer(cellStr, " ");
					row = Long.parseLong( st.nextToken() );
					col = Long.parseLong( st.nextToken() );
					double lvalue = Double.parseDouble( st.nextToken() );
					tmp.quickSetValue((int)row, (int)col, lvalue);
				}
			}
			else
			{
				while( (value=in.readLine())!=null )
				{
					String cellStr = value.toString().trim();							
					StringTokenizer st = new StringTokenizer(cellStr, " ");
					row = Long.parseLong( st.nextToken() );
					col = Long.parseLong( st.nextToken() );
					double lvalue = Double.parseDouble( st.nextToken() );
					tmp.setValueDenseUnsafe((int)row, (int)col, lvalue);
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
