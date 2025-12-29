/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.util;

import java.io.BufferedInputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Writer;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;

import org.apache.hadoop.io.Writable;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.matrix.data.Pair;

public class LocalFileUtils 
{
	public static final int BUFFER_SIZE = 8192;
	public static final int DOUBLE_BUFFERING_MIN = 100*1024;
	
	//unique IDs per JVM for tmp files
	private static IDSequence _seq = null;
	private static String _workingDir = null;
	
	//categories of temp files under process-specific working dir
	public static final String CATEGORY_CACHE        = "cache";
	public static final String CATEGORY_PARTITIONING = "partitioning";
	public static final String CATEGORY_RESULTMERGE  = "resultmerge";
	public static final String CATEGORY_WORK         = "work";
	public static final String CATEGORY_CODEGEN      = "codegen";
	public static final String CATEGORY_LINEAGE      = "lineage";
	
	static {
		_seq = new IDSequence();
	}
	
	/** Reads a matrix block from local file system.
	 * 
	 * @param fname file name to read
	 * @return matrix block
	 * @throws IOException if IOException occurs
	 */
	public static MatrixBlock readMatrixBlockFromLocal(String fname) throws IOException {
		return (MatrixBlock) readWritableFromLocal(fname, new MatrixBlock());
	}
	
	/** Reads a matrix block from local file system.
	 * 
	 * @param fname file name to read
	 * @param reuse matrix block to reuse
	 * @return matrix block
	 * @throws IOException if IOException occurs
	 */
	public static MatrixBlock readMatrixBlockFromLocal(String fname, MatrixBlock reuse) throws IOException {
		return (MatrixBlock) readWritableFromLocal(fname, reuse);
	}

	/** Reads a frame block from local file system.
	 * 
	 * @param fname file name to read
	 * @return frame block
	 * @throws IOException if IOException occurs
	 */
	public static FrameBlock readFrameBlockFromLocal(String fname) throws IOException {
		return (FrameBlock) readWritableFromLocal(fname, new FrameBlock());
	}
	
	/** Reads a matrix/frame block from local file system.
	 * 
	 * @param fname file name to read
	 * @param matrix if true, read matrix. if false, read frame.
	 * @return cache block (common interface to MatrixBlock and FrameBlock)
	 * @throws IOException if IOException occurs
	 */
	public static CacheBlock<?> readCacheBlockFromLocal(String fname, boolean matrix) throws IOException {
		return (CacheBlock<?>) readWritableFromLocal(fname, matrix?new MatrixBlock():new FrameBlock());
	}
	
	/**
	 * Reads an arbitrary writable from local file system, using a fused buffered reader
	 * with special support for matrix blocks.
	 * 
	 * @param fname file name to read
	 * @param ret hadoop writable
	 * @return hadoop writable
	 * @throws IOException if IOException occurs
	 */
	public static Writable readWritableFromLocal(String fname, Writable ret)
		throws IOException
	{
		if(fname == null)
			throw new IOException("Cannot read from null fname");
		FileInputStream fis = new FileInputStream(fname);
		DataInput in = !(ret instanceof MatrixBlock) ? 
			new DataInputStream(new BufferedInputStream(fis, BUFFER_SIZE)) :
			new FastBufferedDataInputStream(fis, BUFFER_SIZE);		
		try {
			ret.readFields(in);
		}
		finally {
			IOUtilFunctions.closeSilently((InputStream)in);
			IOUtilFunctions.closeSilently(fis);
		}
			
		return ret;
	}
	
	/**
	 * Reads a matrix block from an input stream, using a fused buffered reader
	 * with special support for matrix blocks.
	 * 
	 * @param is input stream to read
	 * @return matrix block
	 * @throws IOException if IOException occurs
	 */
	public static MatrixBlock readMatrixBlockFromStream(InputStream is) throws IOException {
		return (MatrixBlock) readWritableFromStream(is, new MatrixBlock());
	}
	
	/**
	 * Reads a frame block from an input stream, using a fused buffered reader
	 * with special support for matrix blocks.
	 * 
	 * @param is input stream to read
	 * @return frame block
	 * @throws IOException if IOException occurs
	 */
	public static FrameBlock readFrameBlockFromStream(InputStream is) throws IOException {
		return (FrameBlock) readWritableFromStream(is, new FrameBlock());
	}
	
	/**
	 * Reads an arbitrary writable from an input stream, using a fused buffered reader
	 * with special support for matrix blocks.
	 * 
	 * @param is input stream to read
	 * @param ret hadoop writable
	 * @return hadoop writable
	 * @throws IOException if IOException occurs
	 */
	public static Writable readWritableFromStream(InputStream is, Writable ret)
		throws IOException
	{
		DataInput in = !(ret instanceof MatrixBlock) ? 
			new DataInputStream(new BufferedInputStream(is, BUFFER_SIZE)) :
			new FastBufferedDataInputStream(is, BUFFER_SIZE);		
		try {
			ret.readFields(in);
		}
		finally {
			IOUtilFunctions.closeSilently((InputStream)in);
			IOUtilFunctions.closeSilently(is);
		}
		
		return ret;
	}
	
	/** Writes a matrix block to local file system.
	 * 
	 * @param fname file name to write
	 * @param mb matrix block
	 * @throws IOException if IOException occurs
	 */
	public static void writeMatrixBlockToLocal(String fname, MatrixBlock mb) throws IOException {
		writeWritableToLocal(fname, mb, mb.getInMemorySize()>=DOUBLE_BUFFERING_MIN);
	}
	
	/** Writes a frame block to local file system.
	 * 
	 * @param fname file name to write
	 * @param fb fame block
	 * @throws IOException if IOException occurs
	 */
	public static void writeFrameBlockToLocal(String fname, FrameBlock fb) throws IOException {
		writeWritableToLocal(fname, fb, fb.getInMemorySize()>=DOUBLE_BUFFERING_MIN);
	}

	/** Writes a matrix/frame block to local file system.
	 * 
	 * @param fname file name to write
	 * @param cb cache block (common interface to matrix block and frame block)
	 * @throws IOException if IOException occurs
	 */
	public static void writeCacheBlockToLocal(String fname, CacheBlock<?> cb) throws IOException {
		writeWritableToLocal(fname, cb, cb.getInMemorySize()>=DOUBLE_BUFFERING_MIN);
	}
	
	/**
	 * Writes an arbitrary writable to local file system, using a fused buffered writer
	 * with special support for matrix blocks.
	 * 
	 * @param fname file name to write
	 * @param mb Hadoop writable
	 * @param doubleBuffering overlay serialization and I/O
	 * @throws IOException if IOException occurs
	 */
	public static void writeWritableToLocal(String fname, Writable mb, boolean doubleBuffering) throws IOException {
		OutputStream fout = doubleBuffering ?
			new DoubleBufferingOutputStream(new FileOutputStream(fname), 2, BUFFER_SIZE) :
			new FileOutputStream(fname);
		FastBufferedDataOutputStream dout = new FastBufferedDataOutputStream(fout, BUFFER_SIZE);
		try {
			mb.write(dout);
		}
		finally {
			IOUtilFunctions.closeSilently(dout);
		}
	}

	public static void writeByteArrayToLocal( String fname, byte[] data )
		throws IOException
	{	
		//byte array write via java.nio file channel ~10-15% faster than java.io
		FileChannel channel = null;
		try {
			Path path = Paths.get(fname);
			channel = FileChannel.open(path, StandardOpenOption.CREATE, 
				StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
			channel.write(ByteBuffer.wrap(data));
		}
		finally {
			IOUtilFunctions.closeSilently(channel);
		}
	}

	public static int readBlockSequenceFromLocal(String fname, 
			Pair<MatrixIndexes,MatrixValue>[] outValues, HashMap<MatrixIndexes, Integer> outMap) 
		throws IOException
	{
		FileInputStream fis = new FileInputStream(fname);
		FastBufferedDataInputStream in = new FastBufferedDataInputStream(fis, BUFFER_SIZE);
		int bufferSize = 0;
		
		try {
			int len = in.readInt();
			for( int i=0; i<len; i++ ) {
				outValues[i].getKey().readFields(in);
				outValues[i].getValue().readFields(in);
				if( outMap!=null )
					outMap.put( outValues[i].getKey(), i );
			}
			bufferSize = len;
		}
		finally {
			IOUtilFunctions.closeSilently(in);
			IOUtilFunctions.closeSilently(fis);
		}
			
		return bufferSize;
	}

	public static void writeBlockSequenceToLocal(String fname, Pair<MatrixIndexes,MatrixValue>[] inValues, int len) 
		throws IOException
	{
		if( len > inValues.length )
			throw new IOException("Invalid length of block sequence: len="+len+" vs data="+inValues.length);
		
		FileOutputStream fos = new FileOutputStream(fname);
		FastBufferedDataOutputStream out = new FastBufferedDataOutputStream(fos, BUFFER_SIZE);
		
		try 
		{
			out.writeInt(len);
			for( int i=0; i<len; i++ ) {
				inValues[i].getKey().write(out);
				inValues[i].getValue().write(out);
			}
		}
		finally{
			IOUtilFunctions.closeSilently(out);	
			IOUtilFunctions.closeSilently(fos);
		}	
	}

	public static boolean createLocalFileIfNotExist(String dir) {
		boolean ret = true;		
		File fdir = new File(dir);
		if( !fdir.exists() )
			ret = fdir.mkdirs();
		
		return ret;
	}

	public static void deleteFileIfExists(String dir) {
		deleteFileIfExists(dir, false);
	}

	public static void deleteFileIfExists(String dir, boolean fileOnly) 
	{
		File fdir = new File(dir);
		
		if( fdir.exists() ) 
		{
			if( fileOnly ) //delete single file
				fdir.delete();
			else //recursively delete entire directory
				rDelete(fdir);	
		}
	}

	public static boolean isExisting(String dir) {
		File fdir = new File(dir);
		return fdir.exists();
	}

	public static boolean createLocalFileIfNotExist( String dir, String permission )
	{
		boolean ret = true;
		
		File fdir = new File(dir);
		if( !fdir.exists() ) {
			ret = fdir.mkdirs();
			setLocalFilePermissions(fdir, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
		}
		
		return ret;
	}

	public static void setLocalFilePermissions(File file, String permissions)
	{
		//note: user and group treated the same way
		char[] c = permissions.toCharArray();
		short sU = (short)(c[0]-48);
		short sO = (short)(c[2]-48); 
		
		file.setExecutable( (sU&1)==1, (sO&1)==0 );
		file.setWritable(   (sU&2)==2, (sO&2)==0 );
		file.setReadable(   (sU&4)==4, (sO&4)==0 );
	}

	
	///////////
	// working dir handling
	///

	public static String checkAndCreateStagingDir(String dir) {
		File f =  new File(dir);
		if( !f.exists() )
			f.mkdirs();
		
		return dir;
	}

	public static String createWorkingDirectory() {
		return createWorkingDirectoryWithUUID( DMLScript.getUUID() );
	}

	public static String createWorkingDirectoryWithUUID(String uuid)
	{
		//create local tmp dir if not existing
		String dirRoot = null;
		DMLConfig conf = ConfigurationManager.getDMLConfig();
		if( conf != null ) 
			dirRoot = conf.getTextValue(DMLConfig.LOCAL_TMP_DIR);
		else 
			dirRoot = DMLConfig.getDefaultTextValue(DMLConfig.LOCAL_TMP_DIR);
		
		//create shared staging dir if not existing
		if( !LocalFileUtils.createLocalFileIfNotExist(dirRoot, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION) ){
			throw new DMLRuntimeException("Failed to create non-existing local working directory: "+dirRoot);
		}
		
		//create process specific sub tmp dir
		StringBuilder sb = new StringBuilder();
		sb.append( dirRoot );
		sb.append(Lop.FILE_SEPARATOR);
		sb.append(Lop.PROCESS_PREFIX);
		sb.append( uuid );
		sb.append(Lop.FILE_SEPARATOR);
		_workingDir = sb.toString();
		
		//create process-specific staging dir if not existing
		if( !LocalFileUtils.createLocalFileIfNotExist(_workingDir) ){
			throw new DMLRuntimeException("Failed to create local working directory: "+_workingDir);
		}
		
		return _workingDir;
	}

	public static void cleanupWorkingDirectory() {
		if( _workingDir != null )
			cleanupWorkingDirectory( _workingDir );
	}

	public static void cleanupWorkingDirectory(String dir) {
		File f =  new File(dir);
		if( f.exists() )
			rDelete(f);
	}

	public static int cleanupRcWorkingDirectory(String dir) {
		int ret = 0;		
		File f =  new File(dir);
		if( f.exists() )
			ret += rcDelete(f);
		
		return ret;
	}
	
	/**
	 * Recursively deletes an entire local file system directory.
	 * 
	 * @param dir directory to delete recursively
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
		
		//delete file/dir itself
		dir.delete();
	}
	
	/**
	 * Recursively deletes an entire local file system directory
	 * and returns the number of files deleted.
	 * 
	 * @param dir directory to delete recursively
	 * @return number of files deleted
	 */
	public static int rcDelete(File dir)
	{
		int count = 0;
		
		//recursively delete files if required
		if( dir.isDirectory() )
		{
			File[] files = dir.listFiles();
			for( File f : files )
				count += rcDelete( f );	
		}
		
		//delete file/dir itself
		count += dir.delete() ? 1 : 0;
		
		return count;
	}

	public static String getWorkingDir(String category) {
		if( _workingDir == null )
			createWorkingDirectory();
		StringBuilder sb = new StringBuilder();
		sb.append( _workingDir );
		sb.append( Lop.FILE_SEPARATOR );
		sb.append( category );
		sb.append( Lop.FILE_SEPARATOR );
		return sb.toString();
	}

	public static String getUniqueWorkingDir(String category) {
		if( _workingDir == null )
			createWorkingDirectory();
		StringBuilder sb = new StringBuilder();
		sb.append( _workingDir );
		sb.append( category );
		sb.append( Lop.FILE_SEPARATOR );
		sb.append( "tmp" );
		sb.append( _seq.getNextID() );
		return sb.toString();
	}
	
	/**
	 * Writes a simple text file to local file system.
	 * 
	 * @param file output file
	 * @param text content of text file 
	 * @throws IOException errors in file open/write/close
	 */
	public static void writeTextFile(File file, String text) 
		throws IOException 
	{
		Writer writer = null;
		try {
			writer = new FileWriter(file);
			writer.write(text);
			writer.flush();
		}
		finally {
			IOUtilFunctions.closeSilently(writer);
		}
	}
}
