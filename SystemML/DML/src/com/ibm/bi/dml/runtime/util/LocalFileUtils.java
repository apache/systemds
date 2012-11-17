package com.ibm.bi.dml.runtime.util;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.utils.configuration.DMLConfig;

public class LocalFileUtils 
{
	public static final int BUFFER_SIZE = 8192;
	
	/**
	 * 
	 * @param filePathAndName
	 * @return
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixBlockFromLocal(String filePathAndName)
		throws FileNotFoundException, IOException
	{
		FileInputStream fis = new FileInputStream( filePathAndName );
		BufferedInputStream bis = new BufferedInputStream( fis, BUFFER_SIZE );
		DataInputStream in = new DataInputStream( bis );
	
		MatrixBlock mb = new MatrixBlock ();
		try
		{
			mb.readFields(in);
		}
		finally
		{
			if( in != null )
				in.close();
		}
			
		return mb;
	}
	
	/**
	 * 
	 * @param filePathAndName
	 * @param mb
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static void writeMatrixBlockToLocal (String filePathAndName, MatrixBlock mb)
		throws FileNotFoundException, IOException
	{
		FileOutputStream fos = new FileOutputStream( filePathAndName );
		BufferedOutputStream bos = new BufferedOutputStream( fos, BUFFER_SIZE );
		DataOutputStream out = new DataOutputStream( bos );
		
		try 
		{
			mb.write (out);
		}
		finally
		{
			if( out != null )
				out.close ();	
		}		
	}

	/**
	 * 
	 * @param dir
	 */
	public static void createLocalFileIfNotExist(String dir) 
	{
		File fdir = new File(dir);
		if( !fdir.exists() )
		{
			fdir.mkdirs();
		}
	}
	
	/**
	 * 
	 * @param dir
	 */
	public static void deleteFileIfExists(String dir) 
	{
		File fdir = new File(dir);
		if( fdir.exists() )
			rDelete(fdir);
	}
	

	/**
	 * 
	 * @param dir
	 */
	private static void rDelete(File dir)
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
	
	
	/**
	 * 
	 * @param dir
	 * @param permission
	 */
	public static void createLocalFileIfNotExist( String dir, String permission )
	{
		File fdir = new File(dir);
		if( !fdir.exists() )
		{
			fdir.mkdirs();
			setLocalFilePermissions(fdir, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
		}
	}
	
	/**
	 * 
	 * @param file
	 * @param permissions
	 */
	public static void setLocalFilePermissions( File file, String permissions )
	{
		//note: user and group treated the same way
		char[] c = permissions.toCharArray();
		short sU = (short)(c[0]-48);
		short sO = (short)(c[2]-48); 
		
		file.setExecutable( (sU&1)==1, (sO&1)==0 );
		file.setWritable(   (sU&2)==2, (sO&2)==0 );
		file.setReadable(   (sU&4)==4, (sO&4)==0 );
	}

	
}
