package com.ibm.bi.dml.runtime.controlprogram.parfor.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.LinkedList;
import java.util.StringTokenizer;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;

public class StagingFileUtils 
{
	
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
			for( Cell c : buffer )
			{
				StringBuilder sb = new StringBuilder();
				sb.append(c.row);
				sb.append(" ");
				sb.append(c.col);
				sb.append(" ");
				sb.append(c.value);
				sb.append("\n");
				out.write( sb.toString() );
			}
		}
		finally
		{
			if( out != null )
				out.close();	
		}	
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
		throws IOException 
	{
		MatrixBlock tmp = new MatrixBlock( brlen, bclen, false );
		tmp.spaceAllocForDenseUnsafe(brlen, bclen);
		
		FileInputStream fis = new FileInputStream( fname );
		BufferedReader in = new BufferedReader(new InputStreamReader(fis));	
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
				tmp.setValueDenseUnsafe((int)row, (int)col, lvalue);
			}
		}
		finally
		{
			if( in != null )
				in.close();
		}
			
		return tmp;
	}
	
	
	/**
	 * 
	 * @param dir
	 * @return
	 */
	public static String checkAndCreateStagingDir(String dir) 
	{
		File f =  new File(dir);		
		if( !f.exists() )
			f.mkdirs();
		
		return dir;
	}
	
	/**
	 * 
	 * @param dir
	 * @return
	 */
	public static String cleanupStagingDir(String dir) 
	{
		File f =  new File(dir);
		if( f.exists() )
			rDelete(f);
		
		return dir;
	}

	
	/**
	 * 
	 * @param dir
	 */
	public static void rDelete(File dir)
	{
		//recursively delete files if required
		if( dir.isDirectory() )
		{
			File[] files = dir.listFiles();
			for( File f : files )
				rDelete( f );	
		}
		
		//delete file itself
		dir.delete();
	}
}
