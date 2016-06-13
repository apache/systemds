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
import java.util.List;

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
import org.apache.wink.json4j.OrderedJSONObject;


public class MapReduceTool 
{
	private static final Log LOG = LogFactory.getLog(MapReduceTool.class.getName());
	private static JobConf _rJob = null; //cached job conf for read-only operations
	
	static{
		_rJob = ConfigurationManager.getCachedJobConf();
	}
	
	
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
	
	@Deprecated
	public static String getUniqueKeyPerTaskWithLeadingZros(JobConf job, boolean inMapper) {
		String nodePrefix = job.get(MRConfigurationNames.MR_TASK_ATTEMPT_ID);
		int i;
		if (inMapper)
			i = nodePrefix.indexOf("_m_");
		else
			i = nodePrefix.indexOf("_r_");
		int j = nodePrefix.lastIndexOf("_");
		nodePrefix = nodePrefix.substring(i + 3, j);
		return nodePrefix;
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

	public static boolean existsFileOnHDFS(String fname){
		boolean ret = true;
		try{
			Path outpath = new Path(fname);
			ret = FileSystem.get(_rJob).exists(outpath);
		}
		catch(Exception ex)
		{
			LOG.error("Exception caught in existsFileOnHDFS", ex);
			ret = false;
		}
		return ret;
	}
	
	public static void deleteFileIfExistOnHDFS(Path outpath, JobConf job) throws IOException {
		if (FileSystem.get(job).exists(outpath)) {
			FileSystem.get(job).delete(outpath, true);
		}
	}
	
	public static void deleteFileIfExistOnLFS(Path outpath, JobConf job) throws IOException {
		if (FileSystem.getLocal(job).exists(outpath)) {
			FileSystem.getLocal(job).delete(outpath, true);
		}
	}

	public static void deleteFileWithMTDIfExistOnHDFS(String fname)  throws IOException {
		deleteFileIfExistOnHDFS(fname);
		deleteFileIfExistOnHDFS(fname + ".mtd");
	}
	
	public static void deleteFileIfExistOnHDFS(String dir) throws IOException {
		Path outpath = new Path(dir);
		FileSystem fs = FileSystem.get(_rJob);
		if (fs.exists(outpath)) {
			//System.err.println("Deleting " + outpath + " ... ");
			fs.delete(outpath, true);
		}
	}

	public static boolean isHDFSDirectory(String dir) throws IOException {
		FileSystem fs = FileSystem.get(_rJob);
		Path pth = new Path(dir);
		FileStatus fstat = fs.getFileStatus(pth);
		return fstat.isDirectory();
	}

	public static boolean isHDFSFileEmpty(String dir) throws IOException {
		FileSystem fs = FileSystem.get(_rJob);
		return isFileEmpty(fs, dir);
	}

	public static boolean isFileEmpty(FileSystem fs, String dir) throws IOException {
		Path pth = new Path(dir);
		FileStatus fstat = fs.getFileStatus(pth);

		if (fstat.isDirectory()) {
			// it is a directory
			FileStatus[] stats = fs.listStatus(pth);
			if (stats != null) {
				for (FileStatus stat : stats) {
					if (stat.getLen() > 0)
						return false;
				}
				return true;
			} else {
				return true;
			}
		} else {
			// it is a regular file
			if (fstat.getLen() == 0)
				return true;
			else
				return false;
		}
	}

	public static void renameFileOnHDFS(String originalDir, String newDir) throws IOException {
		Path originalpath = new Path(originalDir);
		
		deleteFileIfExistOnHDFS(newDir);
		Path newpath = new Path(newDir);
		
		FileSystem fs = FileSystem.get(_rJob);
		if (fs.exists(originalpath)) {
			fs.rename(originalpath, newpath);
		}
		else {
			throw new FileNotFoundException(originalDir);
		}
	}
	
	public static void mergeIntoSingleFile(String originalDir, String newFile) throws IOException {
		FileSystem fs = FileSystem.get(_rJob);
		FileUtil.copyMerge(fs, new Path(originalDir), fs, new Path(newFile), true, _rJob, null);
	}

	public static void copyFileOnHDFS(String originalDir, String newDir) throws IOException {
		Path originalPath = new Path(originalDir);
		Path newPath = new Path(newDir);
		boolean deleteSource = false;
		boolean overwrite = true;
		
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		FileSystem fs = FileSystem.get(job);
		if (fs.exists(originalPath)) {
			FileUtil.copy(fs, originalPath, fs, newPath, deleteSource, overwrite, job);
		}
	}

	/**
	 * 
	 * @param dir
	 * @return
	 * @throws IOException
	 */
	public static String getSubDirs(String dir) 
		throws IOException 
	{
		FileSystem fs = FileSystem.get(_rJob); 
		FileStatus[] files = fs.listStatus(new Path(dir));
		StringBuilder sb = new StringBuilder();
		for (FileStatus file : files) {
			if ( sb.length()>0 )
				sb.append(",");
			sb.append(file.getPath().toString());
		}
		return sb.toString();
	}

	/**
	 * 
	 * @param dir
	 * @return
	 * @throws IOException
	 */
	public static String getSubDirsIgnoreLogs(String dir) 
		throws IOException 
	{
		FileSystem fs = FileSystem.get(_rJob);
		FileStatus[] files = fs.listStatus(new Path(dir));
		StringBuilder sb = new StringBuilder();
		for (FileStatus file : files) {
			String name = file.getPath().toString();
			if (name.contains("_logs"))
				continue;
			if( sb.length()>0 )
				sb.append(",");
			sb.append(name);
		}
		return sb.toString();
	}
	
	/**
	 * Returns the size of a file or directory on hdfs in bytes.
	 * 
	 * @param path
	 * @return
	 * @throws IOException
	 */
	public static long getFilesizeOnHDFS( Path path ) 
		throws IOException
	{
		FileSystem fs = FileSystem.get(_rJob);
		long ret = 0; //in bytes
		if( fs.isDirectory(path) )
			ret = fs.getContentSummary(path).getLength();
		else
			ret = fs.getFileStatus(path).getLen();
		//note: filestatus would return 0 on directories
		
		return ret;
	}
	
	private static BufferedReader setupInputFile ( String filename ) throws IOException {
        Path pt=new Path(filename);
        FileSystem fs = FileSystem.get(_rJob);
        BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(pt)));		
        return br;
	}
	
	public static double readDoubleFromHDFSFile(String filename) 
		throws IOException 
	{
		BufferedReader br = setupInputFile(filename);
		String line = br.readLine();
		br.close();
		if( line == null )
			throw new IOException("Empty file on hdfs: "+filename);
		return Double.parseDouble(line);
	}
	
	public static long readIntegerFromHDFSFile(String filename) 
		throws IOException 
	{
		BufferedReader br = setupInputFile(filename);
		String line = br.readLine();
		br.close();
		if( line == null )
			throw new IOException("Empty file on hdfs: "+filename);
		return Long.parseLong(line);
	}
	
	public static boolean readBooleanFromHDFSFile(String filename) 
		throws IOException 
	{
		BufferedReader br = setupInputFile(filename);
		String line = br.readLine();
		br.close();
		if( line == null )
			throw new IOException("Empty file on hdfs: "+filename);
		return Boolean.parseBoolean(line);
	}
	
	public static String readStringFromHDFSFile(String filename) 
		throws IOException 
	{
		BufferedReader br = setupInputFile(filename);
		// handle multi-line strings in the HDFS file
		StringBuilder sb = new StringBuilder();
		String line = null;
		while ( (line = br.readLine()) != null ) {
			sb.append(line);
			sb.append("\n");
		}
		br.close();
		
		//return string without last character
		return sb.substring(0, sb.length()-1);
	}
		
	private static BufferedWriter setupOutputFile ( String filename ) throws IOException {
        Path pt=new Path(filename);
        FileSystem fs = FileSystem.get(_rJob);
        BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));		
        return br;
	}
	
	public static void writeDoubleToHDFS ( double d, String filename ) throws IOException {
        BufferedWriter br = setupOutputFile(filename);
        String line = "" + d;
        br.write(line);
        br.close();
	}
	
	public static void writeIntToHDFS ( long i, String filename ) throws IOException {
        BufferedWriter br = setupOutputFile(filename);
        String line = "" + i;
        br.write(line);
        br.close();
	}
	
	public static void writeBooleanToHDFS ( boolean b, String filename ) throws IOException {
        BufferedWriter br = setupOutputFile(filename);
        String line = "" + b;
        br.write(line);
        br.close();
	}
	
	public static void writeStringToHDFS ( String s, String filename ) throws IOException {
        BufferedWriter br = setupOutputFile(filename);
        String line = "" + s;
        br.write(line);
        br.close();
	}
	
	public static void writeDimsFile ( String filename, byte[] unknownFlags, long[] maxRows, long[] maxCols) throws IOException {
        BufferedWriter br = setupOutputFile(filename);
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
        br.close();
        //System.out.println("Finished writing dimsFile: " + filename);
	}
	
	public static MatrixCharacteristics[] processDimsFiles(String dir, MatrixCharacteristics[] stats) 
		throws IOException 
	{
		Path pt=new Path(dir);
        FileSystem fs = FileSystem.get(_rJob);
		
        if ( !fs.exists(pt) )
        	return stats;
        
        FileStatus fstat = fs.getFileStatus(pt);
		
        if ( fstat.isDirectory() ) 
        {
			FileStatus[] files = fs.listStatus(pt);
			for ( int i=0; i < files.length; i++ ) {
				Path filePath = files[i].getPath();
				//System.out.println("Processing dims file: " + filePath.toString());
				BufferedReader br = setupInputFile(filePath.toString());
				
				String line = "";
				while((line=br.readLine()) != null ) {
					String[] parts = line.split(" ");
					int resultIndex = Integer.parseInt(parts[0]);
					long maxRows = Long.parseLong(parts[1]);
					long maxCols = Long.parseLong(parts[2]);
					
					stats[resultIndex].setDimension( (stats[resultIndex].getRows() < maxRows ? maxRows : stats[resultIndex].getRows()), 
							                         (stats[resultIndex].getCols() < maxCols ? maxCols : stats[resultIndex].getCols()) );
				}
				
				br.close();
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
	
	public static void writeMetaDataFile(String mtdfile, ValueType vt, List<ValueType> schema, DataType dt, MatrixCharacteristics mc, OutputInfo outinfo) 
		throws IOException {
		writeMetaDataFile(mtdfile, vt, schema, dt, mc, outinfo, null);
	}

	public static void writeMetaDataFile(String mtdfile, ValueType vt, MatrixCharacteristics mc,  OutputInfo outinfo, FileFormatProperties formatProperties) 
		throws IOException {
		writeMetaDataFile(mtdfile, vt, null, DataType.MATRIX, mc, outinfo, formatProperties);
	}
	
	public static void writeMetaDataFile(String mtdfile, ValueType vt, List<ValueType> schema, DataType dt, MatrixCharacteristics mc, 
			OutputInfo outinfo, FileFormatProperties formatProperties) 
		throws IOException 
	{
		Path pt = new Path(mtdfile);
		FileSystem fs = FileSystem.get(_rJob);
		BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
		formatProperties = (formatProperties==null && outinfo==OutputInfo.CSVOutputInfo) ?
				new CSVFileFormatProperties() : formatProperties;
		OrderedJSONObject mtd = new OrderedJSONObject(); // maintain order in output file

		try {
			// build JSON metadata object
			mtd.put(DataExpression.DATATYPEPARAM, dt.toString().toLowerCase());
			if (schema == null)
				switch (vt) {
					case DOUBLE:
						mtd.put(DataExpression.VALUETYPEPARAM, "double");
						break;
					case INT:
						mtd.put(DataExpression.VALUETYPEPARAM, "int");
						break;
					case BOOLEAN:
						mtd.put(DataExpression.VALUETYPEPARAM, "boolean");
						break;
					case STRING:
						mtd.put(DataExpression.VALUETYPEPARAM, "string");
						break;
					case UNKNOWN:
						mtd.put(DataExpression.VALUETYPEPARAM, "unknown");
						break;
					case OBJECT:
						mtd.put(DataExpression.VALUETYPEPARAM, "object");
						break;
				}
			else
			{
				StringBuffer schemaStrBuffer = new StringBuffer();
				for(int i=0; i < schema.size(); i++) {
					switch (schema.get(i)) {
						case DOUBLE:
							schemaStrBuffer.append("DOUBLE");
							break;
						case INT:
							schemaStrBuffer.append("INT");
							break;
						case BOOLEAN:
							schemaStrBuffer.append("BOOLEAN");
							break;
						case STRING:
							schemaStrBuffer.append("STRING");
							break;
						case UNKNOWN:
						default:
							schemaStrBuffer.append("*");
							break;
					}
					schemaStrBuffer.append(DataExpression.DEFAULT_DELIM_DELIMITER);
				}
				mtd.put(DataExpression.SCHEMAPARAM, schemaStrBuffer.toString());
			}
			mtd.put(DataExpression.READROWPARAM, mc.getRows());
			mtd.put(DataExpression.READCOLPARAM, mc.getCols());
			// only output rows_in_block and cols_in_block for matrix binary format
			if (outinfo == OutputInfo.BinaryBlockOutputInfo && dt.isMatrix() ) {
				mtd.put(DataExpression.ROWBLOCKCOUNTPARAM, mc.getRowsPerBlock());
				mtd.put(DataExpression.COLUMNBLOCKCOUNTPARAM, mc.getColsPerBlock());
			}
			// only output nnz for matrix
			if( dt.isMatrix() ) {
				mtd.put(DataExpression.READNUMNONZEROPARAM, mc.getNonZeros());
			}
			if (outinfo == OutputInfo.TextCellOutputInfo) {
				mtd.put(DataExpression.FORMAT_TYPE, "text");
			} else if (outinfo == OutputInfo.BinaryBlockOutputInfo || outinfo == OutputInfo.BinaryCellOutputInfo || outinfo == OutputInfo.BinaryBlockFrameOutputInfo) {
				mtd.put(DataExpression.FORMAT_TYPE, "binary");
			} else if (outinfo == OutputInfo.CSVOutputInfo) {
				mtd.put(DataExpression.FORMAT_TYPE, "csv");
			} else {
				mtd.put(DataExpression.FORMAT_TYPE, "specialized");
			}
			if (outinfo == OutputInfo.CSVOutputInfo) {
				CSVFileFormatProperties csvProperties = (CSVFileFormatProperties) formatProperties;
				mtd.put(DataExpression.DELIM_HAS_HEADER_ROW, csvProperties.hasHeader());
				mtd.put(DataExpression.DELIM_DELIMITER, csvProperties.getDelim());
			}
			mtd.put(DataExpression.DESCRIPTIONPARAM,
					new OrderedJSONObject().put(DataExpression.AUTHORPARAM, "SystemML"));

			// write metadata JSON object to file
			mtd.write(br, 4); // indent with 4 spaces
			br.close();
		} catch (Exception e) {
			throw new IOException("Error creating and writing metadata JSON file", e);
		}
	}

	public static void writeScalarMetaDataFile(String mtdfile, ValueType v) throws IOException {
		Path pt=new Path(mtdfile);
		FileSystem fs = FileSystem.get(_rJob);
		BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
		OrderedJSONObject mtd = new OrderedJSONObject(); // maintain order in output file

		try {
			// build JSON metadata object
			mtd.put(DataExpression.DATATYPEPARAM, "scalar");
			switch (v) {
				case DOUBLE:
					mtd.put(DataExpression.VALUETYPEPARAM, "double");
					break;
				case INT:
					mtd.put(DataExpression.VALUETYPEPARAM, "int");
					break;
				case BOOLEAN:
					mtd.put(DataExpression.VALUETYPEPARAM, "boolean");
					break;
				case STRING:
					mtd.put(DataExpression.VALUETYPEPARAM, "string");
					break;
				case UNKNOWN:
					mtd.put(DataExpression.VALUETYPEPARAM, "unknown");
					break;
				case OBJECT:
					throw new IOException("Write of generic object types not supported.");
			}
			mtd.put(DataExpression.FORMAT_TYPE, "text");
			mtd.put(DataExpression.DESCRIPTIONPARAM,
					new OrderedJSONObject().put(DataExpression.AUTHORPARAM, "SystemML"));

			// write metadata JSON object to file
			mtd.write(br, 4); // indent with 4 spaces
			br.close();
		} catch (Exception e) {
			throw new IOException("Error creating and writing metadata JSON file", e);
		}
	}
	
	public static double[][] readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen) 
		throws IOException, DMLRuntimeException
	{
		MatrixReader reader = MatrixReaderFactory.createMatrixReader(inputinfo);
		MatrixBlock mb = reader.readMatrixFromHDFS(dir, rlen, clen, brlen, bclen, rlen*clen);
		return DataConverter.convertToDoubleMatrix(mb);
	}
	
	public static double[] readColumnVectorFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen) 
		throws IOException, DMLRuntimeException
	{
		MatrixReader reader = MatrixReaderFactory.createMatrixReader(inputinfo);
		MatrixBlock mb = reader.readMatrixFromHDFS(dir, rlen, clen, brlen, bclen, rlen*clen);
		return DataConverter.convertToDoubleVector(mb);
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
		
		FileSystem fs=FileSystem.get(_rJob);
		Path path=new Path(dir);
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
		FSDataInputStream currentStream=fs.open(fileToRead, buffsz);
	    DoubleWritable readKey=new DoubleWritable();
	    IntWritable readValue=new IntWritable();
	    
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
	    
	    double ret = readKey.get();
	    if(average) {
	    	if(numRead<=offset+1) {
	    		reader.readNextKeyValuePairs(readKey, readValue);
				cum_weight += readValue.get();
				ret = (ret+readKey.get())/2;
	    	}
	    }
	    currentStream.close();
		return new double[] {ret, (average ? -1 : readValue.get()), (average ? -1 : cum_weight)};
	}
	
	/**
	 * 
	 * @param name
	 * @return
	 */
	public static int extractNumberFromOutputFile(String name)
	{
		int i=name.indexOf("part-");
		assert(i>=0);
		return Integer.parseInt(name.substring(i+5));
	}
	
	/**
	 * 
	 * @param dir
	 * @param permissions
	 * @throws IOException
	 */
	public static void createDirIfNotExistOnHDFS(String dir, String permissions) 
		throws IOException
	{
		Path path = new Path(dir);
		try {
			FileSystem fs = FileSystem.get(_rJob);
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
	
	
	/**
	 * 
	 * @param filename
	 * @param overwrite
	 * @return
	 * @throws IOException
	 */
	public static FSDataOutputStream getHDFSDataOutputStream(String filename, boolean overwrite) 
		throws IOException
	{
		FileSystem fs = FileSystem.get(_rJob);
		Path path = new Path(filename);
		return fs.create(path, overwrite);
	}
}
