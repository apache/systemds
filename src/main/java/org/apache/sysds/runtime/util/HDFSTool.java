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

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.text.StringEscapeUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObjectFactory;
import org.apache.sysds.runtime.io.BinaryBlockSerialization;
import org.apache.sysds.runtime.io.FileFormatProperties;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.io.MatrixReaderFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.OrderedJSONObject;


public class HDFSTool 
{
	private static final int MAX_DELETE_RETRIES = 10;
	
	// non-deprecated properties
	public static final String DFS_REPLICATION = "dfs.replication"; // hdfs-default.xml
	public static final String IO_FILE_BUFFER_SIZE = "io.file.buffer.size"; // core-default.xml
	public static final String IO_SERIALIZATIONS = "io.serializations"; // core-default.xml
	
	// deprecated properties replaced by new props, new prop names used for constants
	// see https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-common/DeprecatedProperties.html
	public static final String DFS_BLOCKSIZE = "dfs.blocksize"; // hdfs-default.xml
	// public static final String DFS_DATANODE_DATA_DIR; // hdfs-default.xml - currently not used
	// public static final String DFS_METRICS_SESSION_ID; // N/A - currently not used
	public static final String DFS_PERMISSIONS_ENABLED = "dfs.permissions.enabled"; // hdfs-default.xml
	public static final String FS_DEFAULTFS = "fs.defaultFS"; // core-default.xml

	//internal param: custom deserializer/serializer (usually 30% faster than WritableSerialization)
	public static final boolean USE_BINARYBLOCK_SERIALIZATION = true;
	
	private static final Log LOG = LogFactory.getLog(HDFSTool.class.getName());

	public static void addBinaryBlockSerializationFramework( Configuration job ) {
		String frameworkList = job.get(IO_SERIALIZATIONS);
		String frameworkClassBB = BinaryBlockSerialization.class.getCanonicalName();
		job.set(IO_SERIALIZATIONS, frameworkClassBB+","+frameworkList);
	}
	
	public static boolean existsFileOnHDFS(String fname) {
		//robustness for empty strings (e.g., JMLC, MLContext)
		if( fname == null || fname.isEmpty() || fname.trim().isEmpty() )
			return false;
		
		try {
			Path path = new Path(fname);
			return IOUtilFunctions.getFileSystem(path).exists(path);
		}
		catch(Exception ex) {
			LOG.error("Failed check existsFileOnHDFS.", ex);
		}
		return false;
	}
	
	public static boolean isDirectory(String fname) {
		//robustness for empty strings (e.g., JMLC, MLContext)
		if( fname == null || fname.isEmpty() || fname.trim().isEmpty() )
			return false;
		
		try {
			Path path = new Path(fname);
			return isDirectory(IOUtilFunctions.getFileSystem(path), path);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException("Failed to check if file is directory", ex);
		}
	}
	
	public static boolean isDirectory(FileSystem fs, Path path) throws IOException {
		return fs.getFileStatus(path).isDirectory();
	}
	
	public static FileStatus[] getDirectoryListing(String fname) {
		try {
			Path path = new Path(fname);
			return IOUtilFunctions
				.getFileSystem(path).listStatus(path);
		}
		catch(Exception ex) {
			LOG.error("Failed listing of directory contents.", ex);
		}
		return new FileStatus[0];
	}

	public static void deleteFileWithMTDIfExistOnHDFS(String fname)  throws IOException {
		deleteFileIfExistOnHDFS(fname);
		deleteFileIfExistOnHDFS(fname + ".mtd");
	}
	
	public static void deleteFileIfExistOnHDFS(String dir) throws IOException {
		Path path = new Path(dir);
		deleteFileIfExists(IOUtilFunctions.getFileSystem(path), path);
	}

	public static void deleteFileIfExistOnHDFS(Path outpath, JobConf job) throws IOException {
		deleteFileIfExists(IOUtilFunctions.getFileSystem(outpath, job), outpath);
	}
	
	public static void deleteFileIfExistOnLFS(Path outpath, JobConf job) throws IOException {
		deleteFileIfExists(FileSystem.getLocal(job), outpath);
	}
	
	private static void deleteFileIfExists(FileSystem fs, Path outpath) throws IOException {
		if( fs.exists(outpath) ) {
			int retries = MAX_DELETE_RETRIES;
			while( !fs.delete(outpath, true) && retries > 0 ) {
				retries--;
			}
		}
	}

	public static boolean isHDFSFileEmpty(String dir) throws IOException {
		//robustness for empty strings (e.g., JMLC, MLContext)
		if( dir == null || dir.isEmpty() || dir.trim().isEmpty())
			return false;
		Path path = new Path(dir);
		FileSystem fs = IOUtilFunctions.getFileSystem(path);
		return isFileEmpty(fs, path);
	}

	public static boolean isFileEmpty(FileSystem fs, Path dir) throws IOException {
		if(fs instanceof LocalFileSystem) {
			// use local Java filesystem, this is much faster.
			java.io.File f = new java.io.File(dir.toString());
			if(f.exists()){

				if(f.isDirectory()) {
					java.io.File[] fff = f.listFiles();
					if(fff.length == 0)
						return false;
					for(File ff : fff) {
						if(Files.size(ff.toPath()) > 0)
							return false;
					}
					return true;
				}
				else
					return Files.size(f.toPath()) <= 0;
			}
			else return false;
		}
		else{
			FileStatus fstat = fs.getFileStatus(dir);

			if(fstat.isDirectory() || IOUtilFunctions.isObjectStoreFileScheme(dir)) {
				// it is a directory
				FileStatus[] stats = fs.listStatus(dir);
				if(stats != null) {
					for(FileStatus stat : stats) {
						if(stat.getLen() > 0)
							return false;
					}
					return true;
				}
				else {
					return true;
				}
			}
			else {
				// it is a regular file
				return fstat.getLen() == 0;
			}
		}
	}

	public static void renameFileOnHDFS(String originalDir, String newDir) throws IOException {
		Path pathOrig = new Path(originalDir);
		Path pathNew = new Path(newDir);
		if( !IOUtilFunctions.isSameFileScheme(pathOrig, pathNew) )
			throw new IOException("Cannot rename files to different target file system.");
		
		deleteFileIfExistOnHDFS(newDir);
		FileSystem fs = IOUtilFunctions.getFileSystem(pathOrig);
		if( fs.exists(pathOrig) )
			fs.rename(pathOrig, pathNew);
		else
			throw new FileNotFoundException(originalDir);
	}
	
	public static void mergeIntoSingleFile(String originalDir, String newFile) throws IOException {
		Path pathOrig = new Path(originalDir);
		Path pathNew = new Path(newFile);
		if( !IOUtilFunctions.isSameFileScheme(pathOrig, pathNew) )
			throw new IOException("Cannot merge files into different target file system.");
		FileSystem fs = IOUtilFunctions.getFileSystem(pathOrig);
		copyMerge(fs, pathOrig, fs, pathNew, true, 
			ConfigurationManager.getCachedJobConf(), null);
	}

	public static void copyFileOnHDFS(String originalDir, String newDir) throws IOException {
		Path originalPath = new Path(originalDir);
		Path newPath = new Path(newDir);
		boolean deleteSource = false;
		boolean overwrite = true;
		
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		FileSystem fs = IOUtilFunctions.getFileSystem(originalPath, job);
		if (fs.exists(originalPath)) {
			FileUtil.copy(fs, originalPath, fs, newPath, deleteSource, overwrite, job);
		}
	}
	
	public static long estimateNnzBasedOnFileSize(Path path,
		long rlen, long clen, int blen, double factor) throws IOException
	{
		return (long) Math.min(rlen*clen, rlen*clen*(getFilesizeOnHDFS(path)/factor/
			OptimizerUtils.estimatePartitionedSizeExactSparsity(rlen, clen, blen, 1.0)));
	}

	/**
	 * Returns the size of a file or directory on hdfs in bytes.
	 * 
	 * @param path file system path
	 * @return file size
	 * @throws IOException if IOException occurs
	 */
	public static long getFilesizeOnHDFS( Path path ) 
		throws IOException
	{
		FileSystem fs = IOUtilFunctions.getFileSystem(path);
		long ret = 0; //in bytes
		if( fs.getFileStatus(path).isDirectory() )
			ret = fs.getContentSummary(path).getLength();
		else
			ret = fs.getFileStatus(path).getLen();
		//note: filestatus would return 0 on directories
		
		return ret;
	}
	
	private static BufferedReader setupInputFile(String filename) throws IOException {
		Path path = new Path(filename);
		FileSystem fs = IOUtilFunctions.getFileSystem(path);
		BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));
		return br;
	}
	
	public static double readDoubleFromHDFSFile(String filename) throws IOException {
		return (Double)readObjectFromHDFSFile(filename, ValueType.FP64);
	}
	
	public static long readIntegerFromHDFSFile(String filename) throws IOException {
		return (Long)readObjectFromHDFSFile(filename, ValueType.INT64);
	}
	
	public static boolean readBooleanFromHDFSFile(String filename) throws IOException {
		return (Boolean)readObjectFromHDFSFile(filename, ValueType.BOOLEAN);
	}
	
	public static String readStringFromHDFSFile(String filename) 
		throws IOException 
	{
		StringBuilder sb = new StringBuilder();
		try( BufferedReader br = setupInputFile(filename) ) {
			// handle multi-line strings in the HDFS file
			String line = null;
			while ( (line = br.readLine()) != null ) {
				sb.append(line);
				sb.append("\n");
			}
		}
		
		//return string without last character
		return sb.substring(0, sb.length()-1);
	}
	
	public static Object readObjectFromHDFSFile(String filename, ValueType vt) throws IOException {
		String line = null;
		try( BufferedReader br = setupInputFile(filename) ) {
			line = br.readLine();
		}
		if( line == null )
			throw new IOException("Empty file on hdfs: "+filename);
		
		switch( vt ) {
			case BOOLEAN: return Boolean.parseBoolean(line);
			case FP64: return Double.parseDouble(line);
			case INT64: return Long.parseLong(line);
			default: return line;
		}
	}
	
	public static ScalarObject readScalarObjectFromHDFSFile(String fname, ValueType vt) {
		try {
			Object obj = null;
			switch( vt ) {
				case INT64:   obj = readIntegerFromHDFSFile(fname); break;
				case FP64:    obj = readDoubleFromHDFSFile(fname); break;
				case BOOLEAN: obj = readBooleanFromHDFSFile(fname); break;
				case STRING:
				default:      obj = readStringFromHDFSFile(fname);
			}
			return ScalarObjectFactory.createScalarObject(vt, obj);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}
	
	private static BufferedWriter setupOutputFile ( String filename ) throws IOException {
		Path path = new Path(filename);
		FileSystem fs = IOUtilFunctions.getFileSystem(path);
		BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));
		return br;
	}
	
	/**
	 * Helper function to write scalars to HDFS,
	 * including writing its meta data and removing CRC files in local file system
	 *
	 * @param scalar scalar data object
	 * @param fname file name
	 */
	public static void writeScalarToHDFS(ScalarObject scalar, String fname) {
		try {
			writeObjectToHDFS(scalar.getValue(), fname);
			writeScalarMetaDataFile(fname +".mtd", scalar.getValueType());

			FileSystem fs = IOUtilFunctions.getFileSystem(fname);
			if (fs instanceof LocalFileSystem) {
				Path path = new Path(fname);
				IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
			}
		}
		catch ( IOException e ) {
			throw new DMLRuntimeException(e);
		}
	}
	
	public static void writeDoubleToHDFS ( double d, String filename ) throws IOException {
		writeObjectToHDFS(d, filename);
	}
	
	public static void writeIntToHDFS ( long i, String filename ) throws IOException {
		writeObjectToHDFS(i, filename);
	}
	
	public static void writeBooleanToHDFS ( boolean b, String filename ) throws IOException {
		writeObjectToHDFS(b, filename);
	}
	
	public static void writeStringToHDFS ( String s, String filename ) throws IOException {
		writeObjectToHDFS(s, filename);
	}
	
	public static void writeObjectToHDFS ( Object obj, String filename ) throws IOException {
		try( BufferedWriter br = setupOutputFile(filename) ) {
			br.write(obj.toString());
		}
	}

	public static void writeMetaDataFile(String mtdfile, ValueType vt, DataCharacteristics mc, FileFormat fmt)
			throws IOException {
		writeMetaDataFile(mtdfile, vt, null, DataType.MATRIX, mc, fmt, null, null);
	}

	public static void writeMetaDataFile(String mtdfile, ValueType vt, ValueType[] schema, DataType dt,
			DataCharacteristics mc, FileFormat fmt)
			throws IOException {
		writeMetaDataFile(mtdfile, vt, schema, dt, mc, fmt, null, null);
	}

	public static void writeMetaDataFile(String mtdfile, ValueType vt, DataCharacteristics dc, FileFormat fmt,
			FileFormatProperties formatProperties)
			throws IOException {
		writeMetaDataFile(mtdfile, vt, null, DataType.MATRIX, dc, fmt, formatProperties, null);
	}

	public static void writeMetaDataFileFrame(String mtdfile, ValueType[] schema, DataCharacteristics dc,
			FileFormat fmt) throws IOException {
		writeMetaDataFile(mtdfile, ValueType.UNKNOWN, schema, DataType.FRAME, dc, fmt, (FileFormatProperties) null,
				null);
	}

	public static void writeMetaDataFile(String mtdfile, ValueType vt, ValueType[] schema, DataType dt,
			DataCharacteristics dc,
			FileFormat fmt, FileFormatProperties formatProperties)
			throws IOException {
		writeMetaDataFile(mtdfile, vt, schema, dt, dc, fmt, formatProperties, null);
	}

	public static void writeMetaDataFile(String mtdfile, ValueType vt, ValueType[] schema, DataType dt,
			DataCharacteristics dc,
			FileFormat fmt, FileFormatProperties formatProperties, String privacyConstraints)
			throws IOException {
		Path path = new Path(mtdfile);
		FileSystem fs = IOUtilFunctions.getFileSystem(path);
		try( BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(path,true))) ) {
			String mtd = metaDataToString(vt, schema, dt, dc, fmt, formatProperties);
			br.write(mtd);
		} catch (Exception e) {
			throw new IOException("Error creating and writing metadata JSON file", e);
		}
	}

	public static void writeScalarMetaDataFile(String mtdfile, ValueType vt) 
		throws IOException 
	{
		Path path = new Path(mtdfile);
		FileSystem fs = IOUtilFunctions.getFileSystem(path);
		try( BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(path,true))) ) {
			String mtd = metaDataToString(vt, null, DataType.SCALAR, null, FileFormat.TEXT, null);
			br.write(mtd);
		} 
		catch (Exception e) {
			throw new IOException("Error creating and writing metadata JSON file", e);
		}
	}

	public static String metaDataToString(ValueType vt, ValueType[] schema, DataType dt,
			DataCharacteristics dc, FileFormat fmt, FileFormatProperties formatProperties)
			throws JSONException, DMLRuntimeException {
		return metaDataToString(vt, schema, dt, dc, fmt, formatProperties, null);
	}

	public static String metaDataToString(ValueType vt, ValueType[] schema, DataType dt,
			DataCharacteristics dc, FileFormat fmt, FileFormatProperties formatProperties, String privacyConstraints)
			throws JSONException, DMLRuntimeException {
		OrderedJSONObject mtd = new OrderedJSONObject(); // maintain order in output file

		//handle data type and value types (incl schema for frames)
		mtd.put(DataExpression.DATATYPEPARAM, dt.toString().toLowerCase());
		if (schema == null) {
			mtd.put(DataExpression.VALUETYPEPARAM, vt.toExternalString().toLowerCase());
		}	
		else {
			StringBuffer schemaSB = new StringBuffer();
			for(int i=0; i < schema.length; i++) {
				if( schema[i] == ValueType.UNKNOWN )
					schemaSB.append("*");
				else
					schemaSB.append(schema[i].toString());
				schemaSB.append(DataExpression.DEFAULT_DELIM_DELIMITER);
			}
			mtd.put(DataExpression.SCHEMAPARAM, schemaSB.toString());
		}
		
		//handle output dimensions
		if( !dt.isScalar() ) {
			mtd.put(DataExpression.READROWPARAM, dc.getRows());
			mtd.put(DataExpression.READCOLPARAM, dc.getCols());
			// handle output nnz and binary block configuration
			if( dt.isMatrix() ) {
				if (fmt == FileFormat.BINARY || fmt == FileFormat.COMPRESSED) {
					mtd.put(DataExpression.ROWBLOCKCOUNTPARAM, dc.getBlocksize());
					mtd.put(DataExpression.COLUMNBLOCKCOUNTPARAM, dc.getBlocksize());
				}
				mtd.put(DataExpression.READNNZPARAM, dc.getNonZeros());
			}
		}
			
		//handle format type and additional arguments
		mtd.put(DataExpression.FORMAT_TYPE, fmt.toString());
		
		if (formatProperties != null) {
			String description = formatProperties.getDescription();
			if (StringUtils.isNotEmpty(description)) {
				String jsonDescription = StringEscapeUtils.escapeJson(description);
				mtd.put(DataExpression.DESCRIPTIONPARAM, jsonDescription);
			}
		}

		//add username and time
		String userName = System.getProperty("user.name");
		if (StringUtils.isNotEmpty(userName)) {
			mtd.put(DataExpression.AUTHORPARAM, userName);
		} else {
			mtd.put(DataExpression.AUTHORPARAM, "SystemDS");
		}
		
		if (formatProperties instanceof FileFormatPropertiesCSV) {
			FileFormatPropertiesCSV csvProps = (FileFormatPropertiesCSV) formatProperties;
			mtd.put(DataExpression.DELIM_HAS_HEADER_ROW, csvProps.hasHeader());
			mtd.put(DataExpression.DELIM_DELIMITER, csvProps.getDelim());
		}

		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss z");
		mtd.put(DataExpression.CREATEDPARAM, sdf.format(new Date()));

		// Add privacy constraints if specified (must be 'private', 'private-aggregate',
		// or 'public')
		if (privacyConstraints != null && !privacyConstraints.trim().isEmpty()) {
			// Validate privacy constraint value
			if (!privacyConstraints.equals("private") &&
					!privacyConstraints.equals("private-aggregate") &&
					!privacyConstraints.equals("public")) {
				throw new DMLRuntimeException("Invalid privacy constraint: " + privacyConstraints
						+ ". Must be 'private', 'private-aggregate', or 'public'.");
			}
			mtd.put(DataExpression.PRIVACY, privacyConstraints);
		}

		return mtd.toString(4); // indent with 4 spaces
	}

	public static double[][] readMatrixFromHDFS(String dir, FileFormat fmt, long rlen, long clen, int blen)
		throws IOException, DMLRuntimeException
	{
		MatrixReader reader = MatrixReaderFactory.createMatrixReader(fmt);
		long estnnz = (rlen <= 0 || clen <= 0) ? -1 : rlen * clen;
		MatrixBlock mb = reader.readMatrixFromHDFS(dir, rlen, clen, blen, estnnz);
		return DataConverter.convertToDoubleMatrix(mb);
	}
	
	public static double[] readColumnVectorFromHDFS(String dir, FileFormat fmt, long rlen, long clen, int blen)
		throws IOException, DMLRuntimeException
	{
		MatrixReader reader = MatrixReaderFactory.createMatrixReader(fmt);
		long estnnz = (rlen <= 0 || clen <= 0) ? -1 : rlen * clen;
		MatrixBlock mb = reader.readMatrixFromHDFS(dir, rlen, clen, blen, estnnz);
		return DataConverter.convertToDoubleVector(mb, false);
	}

	public static void createDirIfNotExistOnHDFS(Path path) throws IOException {
		createDirIfNotExistOnHDFS(path, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
	}

	public static void createDirIfNotExistOnHDFS(String dir, String permissions) 
		throws IOException 
	{
		createDirIfNotExistOnHDFS(new Path(dir), permissions);
	}
	
	public static void createDirIfNotExistOnHDFS(Path path, String permissions) 
		throws IOException
	{
		try {
			FileSystem fs = IOUtilFunctions.getFileSystem(path);
			if( !fs.exists(path) ) 
			{
				char[] c = permissions.toCharArray();
				short sU = (short)((c[0]-48) * 64);
				short sG = (short)((c[1]-48) * 8);
				short sO = (short)((c[2]-48)); 
				short mode = (short)(sU + sG + sO);
				FsPermission perm = new FsPermission(mode);
				fs.mkdirs(path, perm);
			}	
		}
		catch (Exception ex){
			throw new IOException("Failed in creating a non existing dir on HDFS", ex);
		}
		
		//NOTE: we depend on the configured umask, setting umask in job or fspermission has no effect
		//similarly setting MRConfigurationNames.DFS_DATANODE_DATA_DIR_PERM as no effect either.
	}

	public static FSDataOutputStream getHDFSDataOutputStream(String filename, boolean overwrite) 
		throws IOException
	{
		Path path = new Path(filename);
		return IOUtilFunctions.getFileSystem(path)
			.create(path, overwrite);
	}
	
	//copy from hadoop 2.x as this method was removed from hadoop 3.x
	private static boolean copyMerge(FileSystem srcFS, Path srcDir,
		FileSystem dstFS, Path dstFile, boolean deleteSource,
		Configuration conf, String addString) throws IOException
	{
		dstFile = checkDest(srcDir.getName(), dstFS, dstFile, false);
		if (!srcFS.getFileStatus(srcDir).isDirectory())
			return false;
		OutputStream out = dstFS.create(dstFile);
		try {
			FileStatus contents[] = srcFS.listStatus(srcDir);
			Arrays.sort(contents);
			for (int i = 0; i < contents.length; i++) {
				if (contents[i].isFile()) {
					InputStream in = srcFS.open(contents[i].getPath());
					try {
						IOUtils.copyBytes(in, out, conf, false);
						if (addString!=null)
							out.write(addString.getBytes("UTF-8"));
					} finally {
						in.close();
					} 
				}
			}
		} finally {
			out.close();
		}
		if (deleteSource) {
			return srcFS.delete(srcDir, true);
		} else {
			return true;
		}
	}
	
	private static Path checkDest(String srcName, FileSystem dstFS, Path dst,
		boolean overwrite) throws IOException {
		if (dstFS.exists(dst)) {
			FileStatus sdst = dstFS.getFileStatus(dst);
			if (sdst.isDirectory()) {
				if (null == srcName) {
					throw new IOException("Target " + dst + " is a directory");
				}
				return checkDest(null, dstFS, new Path(dst, srcName), overwrite);
			} else if (!overwrite) {
				throw new IOException("Target " + dst + " already exists");
			}
		}
		return dst;
	}
}
