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

package org.apache.sysml.runtime.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.text.SimpleDateFormat;
import java.util.Date;

import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang3.StringEscapeUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.parser.DataExpression;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.io.MatrixReader;
import org.apache.sysml.runtime.io.MatrixReaderFactory;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FileFormatProperties;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.NumItemsByEachReducerMetaData;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.mapred.MRConfigurationNames;
import org.apache.sysml.runtime.matrix.sort.ReadWithZeros;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.OrderedJSONObject;


public class MapReduceTool 
{
	private static final int MAX_DELETE_RETRIES = 10;
	
	private static final Log LOG = LogFactory.getLog(MapReduceTool.class.getName());
	
	public static String getUniqueKeyPerTask(JobConf job, boolean inMapper) {
		//TODO: investigate ID pattern, required for parallel jobs
		/*String nodePrefix = job.get(MRConfigurationNames.MR_TASK_ATTEMPT_ID);
		return String.valueOf(IDHandler.extractLongID(nodePrefix));*/
		
		String nodePrefix = job.get(MRConfigurationNames.MR_TASK_ATTEMPT_ID);
		int i;
		if (inMapper)
			i = nodePrefix.indexOf("_m_");
		else
			i = nodePrefix.indexOf("_r_");
		int j = nodePrefix.lastIndexOf("_");
		nodePrefix = nodePrefix.substring(i + 3, j);
		// remove all the leading 0s
		return String.valueOf(Long.parseLong(nodePrefix));
	}

	public static int getUniqueTaskId(JobConf job) {
		//TODO: investigate ID pattern, required for parallel jobs
		/*String nodePrefix = job.get(MRConfigurationNames.MR_TASK_ATTEMPT_ID); 
		return IDHandler.extractIntID(nodePrefix);*/
		
		String nodePrefix = job.get(MRConfigurationNames.MR_TASK_ATTEMPT_ID);
		int j = nodePrefix.lastIndexOf("_");
		int i=nodePrefix.lastIndexOf("_", j-1);
		nodePrefix = nodePrefix.substring(i+1, j);
		// System.out.println("nodePrefix = " + nodePrefix) ;
		return Integer.valueOf(nodePrefix);
	}

	public static String getGloballyUniqueName(JobConf job) {
		return job.get(MRConfigurationNames.MR_TASK_ATTEMPT_ID);
	}

	public static boolean existsFileOnHDFS(String fname) {
		//robustness for empty strings (e.g., JMLC, MLContext)
		if( fname == null || fname.isEmpty() )
			return false;
		
		try {
			Path path = new Path(fname);
			return IOUtilFunctions
				.getFileSystem(path).exists(path);
		}
		catch(Exception ex) {
			LOG.error("Failed check existsFileOnHDFS.", ex);
		}
		return false;
	}
	
	public static boolean isDirectory(String fname) {
		//robustness for empty strings (e.g., JMLC, MLContext)
		if( fname == null || fname.isEmpty() )
			return false;
		
		try {
			Path path = new Path(fname);
			return IOUtilFunctions
				.getFileSystem(path).isDirectory(path);
		}
		catch(Exception ex) {
			LOG.error("Failed check isDirectory.", ex);
		}
		return false;
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
		if( dir == null || dir.isEmpty() )
			return false;
		Path path = new Path(dir);
		FileSystem fs = IOUtilFunctions.getFileSystem(path);
		return isFileEmpty(fs, path);
	}

	public static boolean isFileEmpty(FileSystem fs, Path dir) throws IOException {
		FileStatus fstat = fs.getFileStatus(dir);

		if( fstat.isDirectory() 
			|| IOUtilFunctions.isObjectStoreFileScheme(dir) )
		{
			// it is a directory
			FileStatus[] stats = fs.listStatus(dir);
			if (stats != null) {
				for (FileStatus stat : stats) {
					if (stat.getLen() > 0)
						return false;
				}
				return true;
			} else {
				return true;
			}
		} 
		else {
			// it is a regular file
			return (fstat.getLen() == 0);
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
		FileUtil.copyMerge(fs, pathOrig, fs, pathNew, true, 
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
		if( fs.isDirectory(path) )
			ret = fs.getContentSummary(path).getLength();
		else
			ret = fs.getFileStatus(path).getLen();
		//note: filestatus would return 0 on directories
		
		return ret;
	}
	
	private static BufferedReader setupInputFile ( String filename ) throws IOException {
        Path path = new Path(filename);
        FileSystem fs = IOUtilFunctions.getFileSystem(path);
		BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(path)));		
        return br;
	}
	
	public static double readDoubleFromHDFSFile(String filename) throws IOException {
		return (Double)readObjectFromHDFSFile(filename, ValueType.DOUBLE);
	}
	
	public static long readIntegerFromHDFSFile(String filename) throws IOException {
		return (Long)readObjectFromHDFSFile(filename, ValueType.INT);
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
			case DOUBLE: return Double.parseDouble(line);
			case INT: return Long.parseLong(line);
			default: return line;
		}
	}
		
	private static BufferedWriter setupOutputFile ( String filename ) throws IOException {
        Path path = new Path(filename);
        FileSystem fs = IOUtilFunctions.getFileSystem(path);
		BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));		
        return br;
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
	
	public static void writeDimsFile ( String filename, byte[] unknownFlags, long[] maxRows, long[] maxCols) throws IOException {
		try( BufferedWriter br = setupOutputFile(filename) ) {
	        StringBuilder line = new StringBuilder();
	        for ( int i=0; i < unknownFlags.length; i++ ) {
	        	if ( unknownFlags[i]  != (byte)0 ) {
	        		line.append(i);
	        		line.append(" " + maxRows[i]);
	        		line.append(" " + maxCols[i]);
	        		line.append("\n");
	        	}
	        }
	        br.write(line.toString());
        }
	}
	
	public static MatrixCharacteristics[] processDimsFiles(String dir, MatrixCharacteristics[] stats) 
		throws IOException 
	{
		Path path = new Path(dir);
		FileSystem fs = IOUtilFunctions.getFileSystem(path);
		
        if ( !fs.exists(path) )
        	return stats;
        
        FileStatus fstat = fs.getFileStatus(path);
		
        if ( fstat.isDirectory() ) 
        {
			FileStatus[] files = fs.listStatus(path);
			for ( int i=0; i < files.length; i++ ) {
				Path filePath = files[i].getPath();
				try( BufferedReader br = setupInputFile(filePath.toString()) ) {
					String line = "";
					while((line=br.readLine()) != null ) {
						String[] parts = line.split(" ");
						int resultIndex = Integer.parseInt(parts[0]);
						long maxRows = Long.parseLong(parts[1]);
						long maxCols = Long.parseLong(parts[2]);
						
						stats[resultIndex].setDimension( (stats[resultIndex].getRows() < maxRows ? maxRows : stats[resultIndex].getRows()), 
								                         (stats[resultIndex].getCols() < maxCols ? maxCols : stats[resultIndex].getCols()) );
					}
				}
			}
		}
		else 
		{
			throw new IOException(dir + " is expected to be a folder!");
		}

		return stats;
	}
	
	public static void writeMetaDataFile(String mtdfile, ValueType vt, MatrixCharacteristics mc, OutputInfo outinfo) 
		throws IOException {
		writeMetaDataFile(mtdfile, vt, null, DataType.MATRIX, mc, outinfo);
	}
	
	public static void writeMetaDataFile(String mtdfile, ValueType vt, ValueType[] schema, DataType dt, MatrixCharacteristics mc, OutputInfo outinfo) 
		throws IOException {
		writeMetaDataFile(mtdfile, vt, schema, dt, mc, outinfo, null);
	}

	public static void writeMetaDataFile(String mtdfile, ValueType vt, MatrixCharacteristics mc,  OutputInfo outinfo, FileFormatProperties formatProperties) 
		throws IOException {
		writeMetaDataFile(mtdfile, vt, null, DataType.MATRIX, mc, outinfo, formatProperties);
	}
	
	public static void writeMetaDataFile(String mtdfile, ValueType vt, ValueType[] schema, DataType dt, MatrixCharacteristics mc, 
			OutputInfo outinfo, FileFormatProperties formatProperties) 
		throws IOException 
	{
		Path path = new Path(mtdfile);
		FileSystem fs = IOUtilFunctions.getFileSystem(path);
		try( BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(path,true))) ) {
			String mtd = metaDataToString(vt, schema, dt, mc, outinfo, formatProperties);
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
			String mtd = metaDataToString(vt, null, DataType.SCALAR, null, OutputInfo.TextCellOutputInfo, null);
			br.write(mtd);
		} 
		catch (Exception e) {
			throw new IOException("Error creating and writing metadata JSON file", e);
		}
	}

	public static String metaDataToString(ValueType vt, ValueType[] schema, DataType dt, MatrixCharacteristics mc,
			OutputInfo outinfo, FileFormatProperties formatProperties) throws JSONException, DMLRuntimeException
	{
		OrderedJSONObject mtd = new OrderedJSONObject(); // maintain order in output file

		//handle data type and value types (incl schema for frames)
		mtd.put(DataExpression.DATATYPEPARAM, dt.toString().toLowerCase());
		if (schema == null) {
			mtd.put(DataExpression.VALUETYPEPARAM, vt.toString().toLowerCase());
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
			mtd.put(DataExpression.READROWPARAM, mc.getRows());
			mtd.put(DataExpression.READCOLPARAM, mc.getCols());
			// handle output nnz and binary block configuration
			if( dt.isMatrix() ) {
				if (outinfo == OutputInfo.BinaryBlockOutputInfo ) {
					mtd.put(DataExpression.ROWBLOCKCOUNTPARAM, mc.getRowsPerBlock());
					mtd.put(DataExpression.COLUMNBLOCKCOUNTPARAM, mc.getColsPerBlock());
				}
				mtd.put(DataExpression.READNUMNONZEROPARAM, mc.getNonZeros());
			}
		}
			
		//handle format type and additional arguments	
		mtd.put(DataExpression.FORMAT_TYPE, OutputInfo.outputInfoToStringExternal(outinfo));
		if (outinfo == OutputInfo.CSVOutputInfo) {
			CSVFileFormatProperties csvProperties = (formatProperties==null) ?
				new CSVFileFormatProperties() : (CSVFileFormatProperties)formatProperties;
			mtd.put(DataExpression.DELIM_HAS_HEADER_ROW, csvProperties.hasHeader());
			mtd.put(DataExpression.DELIM_DELIMITER, csvProperties.getDelim());
		}

		if (formatProperties != null) {
			String description = formatProperties.getDescription();
			if (StringUtils.isNotEmpty(description)) {
				String jsonDescription = StringEscapeUtils.escapeJson(description);
				mtd.put(DataExpression.DESCRIPTIONPARAM, jsonDescription);
			}
		}

		String userName = System.getProperty("user.name");
		if (StringUtils.isNotEmpty(userName)) {
			mtd.put(DataExpression.AUTHORPARAM, userName);
		} else {
			mtd.put(DataExpression.AUTHORPARAM, "SystemML");
		}

		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss z");
		mtd.put(DataExpression.CREATEDPARAM, sdf.format(new Date()));

		return mtd.toString(4); // indent with 4 spaces	
	}
	
	public static double[][] readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen) 
		throws IOException, DMLRuntimeException
	{
		MatrixReader reader = MatrixReaderFactory.createMatrixReader(inputinfo);
		long estnnz = (rlen <= 0 || clen <= 0) ? -1 : rlen * clen;
		MatrixBlock mb = reader.readMatrixFromHDFS(dir, rlen, clen, brlen, bclen, estnnz);
		return DataConverter.convertToDoubleMatrix(mb);
	}
	
	public static double[] readColumnVectorFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen) 
		throws IOException, DMLRuntimeException
	{
		MatrixReader reader = MatrixReaderFactory.createMatrixReader(inputinfo);
		long estnnz = (rlen <= 0 || clen <= 0) ? -1 : rlen * clen;
		MatrixBlock mb = reader.readMatrixFromHDFS(dir, rlen, clen, brlen, bclen, estnnz);
		return DataConverter.convertToDoubleVector(mb, false);
	}
	
	public static double median(String dir, NumItemsByEachReducerMetaData metadata) throws IOException {
		long[] counts=metadata.getNumItemsArray();
		long[] ranges=new long[counts.length];
		ranges[0]=counts[0];
		for(int i=1; i<counts.length; i++)
			ranges[i]=ranges[i-1]+counts[i];
		
		long total=ranges[ranges.length-1];
		
		return pickValueWeight(dir, metadata, 0.5, total%2==0)[0];
	}
	

	public static double pickValue(String dir, NumItemsByEachReducerMetaData metadata, double p) throws IOException {
		return pickValueWeight(dir, metadata, p, false)[0];
	}
	
	public static double[] pickValueWeight(String dir, NumItemsByEachReducerMetaData metadata, double p, boolean average) 
	throws IOException
	{
		long[] counts=metadata.getNumItemsArray();
		long[] ranges=new long[counts.length];
		ranges[0]=counts[0];
		for(int i=1; i<counts.length; i++)
			ranges[i]=ranges[i-1]+counts[i];
		
		long total=ranges[ranges.length-1];
		
		// do averaging only if it is asked for; and sum_wt is even
		average = average && (total%2 == 0);

		int currentPart=0;
		double cum_weight = 0;
		long pos=(long)Math.ceil(total*p);
		while(ranges[currentPart]<pos) {
			currentPart++;
			cum_weight += ranges[currentPart];
		}
		int offset;
		if(currentPart>0)
			offset=(int)(pos-ranges[currentPart-1]-1);
		else
			offset=(int)pos-1;
		
		Path path=new Path(dir);
		FileSystem fs=IOUtilFunctions.getFileSystem(path);
		FileStatus[] files=fs.listStatus(path);
		Path fileToRead=null;
		for(FileStatus file: files)
			if(file.getPath().toString().endsWith(Integer.toString(currentPart)))
			{
				fileToRead=file.getPath();
				break;
			}
		
		if(fileToRead==null)
			throw new RuntimeException("cannot read partition "+currentPart);
		
		int buffsz = 64 * 1024;
		DoubleWritable readKey=new DoubleWritable();
	    IntWritable readValue=new IntWritable();
	    FSDataInputStream currentStream = null;
		double ret = -1;
	    try {
			currentStream = fs.open(fileToRead, buffsz);
		    
			boolean contain0s=false;
			long numZeros=0;
			if(currentPart==metadata.getPartitionOfZero())
			{
				contain0s=true;
				numZeros=metadata.getNumberOfZero();
			}
		    ReadWithZeros reader=new ReadWithZeros(currentStream, contain0s, numZeros);
	
		    int numRead=0;
		    while(numRead<=offset)
			{
		    	reader.readNextKeyValuePairs(readKey, readValue);
				numRead+=readValue.get();
				cum_weight += readValue.get();
			}
		    
		    ret = readKey.get();
		    if(average) {
		    	if(numRead<=offset+1) {
		    		reader.readNextKeyValuePairs(readKey, readValue);
					cum_weight += readValue.get();
					ret = (ret+readKey.get())/2;
		    	}
		    }
		}
		finally {
			IOUtilFunctions.closeSilently(currentStream);
		}
	    return new double[] {ret, (average ? -1 : readValue.get()), (average ? -1 : cum_weight)};
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
}
