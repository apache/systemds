/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;

import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.Converter;
import com.ibm.bi.dml.runtime.matrix.data.FileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.NumItemsByEachReducerMetaData;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.data.Pair;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration.ConvertTarget;
import com.ibm.bi.dml.runtime.matrix.sort.ReadWithZeros;


public class MapReduceTool 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
			
	private static final Log LOG = LogFactory.getLog(MapReduceTool.class.getName());
	private static JobConf _rJob = null; //cached job conf for read-only operations
	
	static{
		_rJob = new JobConf();
	}
	
	
	public static String getUniqueKeyPerTask(JobConf job, boolean inMapper) {
		//TODO: investigate ID pattern, required for parallel jobs
		/*String nodePrefix = job.get("mapred.task.id");
		return String.valueOf(IDHandler.extractLongID(nodePrefix));*/
		
		String nodePrefix = job.get("mapred.task.id");
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
		String nodePrefix = job.get("mapred.task.id");
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
		/*String nodePrefix = job.get("mapred.task.id"); 
		return IDHandler.extractIntID(nodePrefix);*/
		
		String nodePrefix = job.get("mapred.task.id");
		int j = nodePrefix.lastIndexOf("_");
		int i=nodePrefix.lastIndexOf("_", j-1);
		nodePrefix = nodePrefix.substring(i+1, j);
		// System.out.println("nodePrefix = " + nodePrefix) ;
		return (new Integer(nodePrefix)).intValue();
	}

	public static String getGloballyUniqueName(JobConf job) {
		return job.get("mapred.task.id");
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
		return fstat.isDir();
	}

	public static boolean isHDFSFileEmpty(String dir) throws IOException {
		FileSystem fs = FileSystem.get(_rJob);
		return isFileEmpty(fs, dir);
	}

	public static boolean isFileEmpty(FileSystem fs, String dir) throws IOException {
		Path pth = new Path(dir);
		FileStatus fstat = fs.getFileStatus(pth);

		if (fstat.isDir()) {
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

	public static void copyFileOnHDFS(String originalDir, String newDir) throws IOException {
		Path originalPath = new Path(originalDir);
		Path newPath = new Path(newDir);
		boolean deleteSource = false;
		boolean overwrite = true;
		
		JobConf job = new JobConf();
		FileSystem fs = FileSystem.get(job);
		if (fs.exists(originalPath)) {
			FileUtil.copy(fs, originalPath, fs, newPath, deleteSource, overwrite, job);
		}
	}

	public static String getSubDirs(String dir) throws IOException {
		FileSystem fs = FileSystem.get(_rJob); 
		FileStatus[] files = fs.listStatus(new Path(dir));
		String ret = "";
		for (FileStatus file : files) {
			if (!ret.isEmpty())
				ret += ",";
			ret += file.getPath().toString();
		}
		return ret;
	}

	public static String getSubDirsIgnoreLogs(String dir) throws IOException {
		FileSystem fs = FileSystem.get(_rJob);
		FileStatus[] files = fs.listStatus(new Path(dir));
		String ret = "";
		for (FileStatus file : files) {
			String name = file.getPath().toString();
			if (name.contains("_logs"))
				continue;
			if (!ret.isEmpty())
				ret += ",";
			ret += name;
		}
		return ret;
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

	public static double readSingleNumberFromHDFS(String dir) throws IOException {
		String filename = getSubDirsIgnoreLogs(dir);
		if (filename.contains(","))
			throw new IOException("expect only one file, but given " + filename);
		Configuration conf = new Configuration();
		SequenceFile.Reader reader = new SequenceFile.Reader(FileSystem.get(conf), new Path(filename), conf);
		MatrixIndexes indexes = new MatrixIndexes();
		MatrixCell value = new MatrixCell();
		if (!reader.next(indexes, value)) {
			reader.close();
			throw new IOException("no item to read!");
		}
		// LOG.info("readSingleNumber from "+filename+": ("+indexes.getRowIndex()+indexes.getColumnIndex()+"): "+value.get());
		assert (indexes.getColumnIndex() == 0 && indexes.getRowIndex() == 0);
		double ret = value.getValue();
		assert (!reader.next(indexes, value));
		reader.close();
		return ret;
	}

	public static double readFirstNumberFromHDFSMatrix(String dir) throws IOException {
		String filename = getSubDirsIgnoreLogs(dir);
		if (filename.contains(","))
			throw new IOException("expect only one file, but given " + filename);
		Configuration conf = new Configuration();
		SequenceFile.Reader reader = new SequenceFile.Reader(FileSystem.get(conf), new Path(filename), conf);
		MatrixIndexes indexes = new MatrixIndexes();
		
		try {
			MatrixBlock value = new MatrixBlock();
			if (!reader.next(indexes, value)) {
				reader.close();
				throw new IOException("no item to read!");
			}
			// LOG.info("readSingleNumber from "+filename+": ("+indexes.getRowIndex()+indexes.getColumnIndex()+"): "+value.get());
			assert (indexes.getColumnIndex() == 0 && indexes.getRowIndex() == 0);

			double ret = value.getValue(0, 0);
			assert (!reader.next(indexes, value));
			reader.close();
			return ret;
		} catch(Exception e) {
			MatrixCell value = new MatrixCell();
			if (!reader.next(indexes, value))
				throw new IOException("no item to read!");
			// LOG.info("readSingleNumber from "+filename+": ("+indexes.getRowIndex()+indexes.getColumnIndex()+"): "+value.get());
			assert (indexes.getColumnIndex() == 0 && indexes.getRowIndex() == 0);

			double ret = value.getValue(0, 0);
			assert (!reader.next(indexes, value));
			reader.close();
			return ret;
		}
	}

	public static double readSingleNumberFromHDFSBlock(String dir) throws IOException {
		String filename = getSubDirsIgnoreLogs(dir);
		if (filename.contains(","))
			throw new IOException("expect only one file, but given " + filename);
		Configuration conf = new Configuration();
		SequenceFile.Reader reader = new SequenceFile.Reader(FileSystem.get(conf), new Path(filename), conf);
		MatrixIndexes indexes = new MatrixIndexes();
		MatrixBlock value = new MatrixBlock();
		if (!reader.next(indexes, value)) {
			reader.close();
			throw new IOException("no item to read!");
		}
		// LOG.info("readSingleNumber from "+filename+": ("+indexes.getRowIndex()+indexes.getColumnIndex()+"): "+value.get());
		assert (indexes.getColumnIndex() == 0 && indexes.getRowIndex() == 0);
		double ret = value.getValue(0, 0);
		assert (!reader.next(indexes, value));
		reader.close();
		return ret;
	}
	
	private static BufferedReader setupInputFile ( String filename ) throws IOException {
        Path pt=new Path(filename);
        FileSystem fs = FileSystem.get(_rJob);
        BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(pt)));		
        return br;
	}
	
	public static double readDoubleFromHDFSFile(String filename) throws IOException {
		BufferedReader br = setupInputFile(filename);
		String line = br.readLine();
		br.close();
		return Double.parseDouble(line);
	}
	public static long readIntegerFromHDFSFile(String filename) throws IOException {
		BufferedReader br = setupInputFile(filename);
		String line = br.readLine();
		br.close();
		return Integer.parseInt(line);
	}
	public static boolean readBooleanFromHDFSFile(String filename) throws IOException {
		BufferedReader br = setupInputFile(filename);
		String line = br.readLine();
		br.close();
		return Boolean.parseBoolean(line);
	}
	public static String readStringFromHDFSFile(String filename) throws IOException {
		BufferedReader br = setupInputFile(filename);
		// handle multi-line strings in the HDFS file
		String output = "", temp = "";
		output = br.readLine();
		while ( (temp = br.readLine()) != null ) {
			output += "\n" + temp;
		}
		br.close();
		return output;
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
	
	public static MatrixCharacteristics[] processDimsFiles(String dir, MatrixCharacteristics[] stats) throws IOException {
		Path pt=new Path(dir);
        FileSystem fs = FileSystem.get(_rJob);
		
        if ( !fs.exists(pt) )
        	return stats;
        
        FileStatus fstat = fs.getFileStatus(pt);
		
        //System.out.println("----------------");
		if ( fstat.isDir() ) {
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
					
					stats[resultIndex].numRows = (stats[resultIndex].numRows < maxRows ? maxRows : stats[resultIndex].numRows);
					stats[resultIndex].numColumns = (stats[resultIndex].numColumns < maxCols ? maxCols : stats[resultIndex].numColumns);
					//System.out.println("     " + line);
				}
				
				br.close();
			}
		}
		else {
			throw new IOException(dir + " is expected to be a folder!");
		}
        //System.out.println("----------------");

		return stats;
	}
	
	public static void writeMetaDataFile ( String mtdfile, ValueType v, MatrixCharacteristics mc, OutputInfo outinfo) throws IOException {
		writeMetaDataFile(mtdfile, v, mc, outinfo, null);
	}
	
	public static void writeMetaDataFile( String mtdfile, ValueType v, MatrixCharacteristics mc, OutputInfo outinfo, FileFormatProperties formatProperties ) 
		throws IOException 
	{
		Path pt = new Path(mtdfile);
        FileSystem fs = FileSystem.get(_rJob);
        BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));		
        formatProperties = (formatProperties==null && outinfo==OutputInfo.CSVOutputInfo) ? 
        		           new CSVFileFormatProperties() : formatProperties;

        String line = "";
        
        try {
          line += "{ \n" +
          "    \"" +  DataExpression.DATATYPEPARAM         +  "\": \"matrix\"\n" +
          "    ,\"" +  DataExpression.VALUETYPEPARAM        +  "\": ";
        
          switch (v) {
          case DOUBLE:
			line += "\"double\"\n";
			break;
	  	  case INT:
			line += "\"int\"\n";
			break;
		  case BOOLEAN:
			line += "\"boolean\"\n";
			break;
		  case STRING:
			line += "\"string\"\n";
			break;
          };
        
          line += 
          "    ,\"" +  DataExpression.READROWPARAM 			+  "\": " + mc.numRows + "\n" + 
		  "    ,\"" + DataExpression.READCOLPARAM 			+  "\": " + mc.numColumns + "\n";
          // only output rows_in_block and cols_in_block for binary format 
          if ( outinfo == OutputInfo.BinaryBlockOutputInfo)  {
         	 line += "    ,\"" + DataExpression.ROWBLOCKCOUNTPARAM	+  "\": " + mc.numRowsPerBlock + "\n" + 
		            "    ,\"" + DataExpression.COLUMNBLOCKCOUNTPARAM +  "\": " + mc.numColumnsPerBlock + "\n";
          }
        
          line += "    ,\"" +	DataExpression.READNUMNONZEROPARAM	+  "\": " + mc.nonZero + "\n" +
		          "    ,\"" + DataExpression.FORMAT_TYPE	+  "\": "; 
        
          if ( outinfo == OutputInfo.TextCellOutputInfo ) {
        	line += "\"text\"\n";
          } else if (outinfo == OutputInfo.BinaryBlockOutputInfo || outinfo == OutputInfo.BinaryCellOutputInfo ) {
        	line += "\"binary\"\n"; // currently, there is no way to differentiate between them
          } else if (outinfo == OutputInfo.CSVOutputInfo ) {
        	line += "\"csv\"\n"; 
          } else {
        	line += "\"specialized\"\n"; 
          }
          
          if ( outinfo == OutputInfo.CSVOutputInfo) {
        	  CSVFileFormatProperties csvProperties = (CSVFileFormatProperties) formatProperties;
              line += "    ,\"" +  DataExpression.DELIM_HAS_HEADER_ROW 	+  "\": " + csvProperties.hasHeader() + "\n";
              line += "    ,\"" +  DataExpression.DELIM_DELIMITER 	+  "\": \"" + csvProperties.getDelim() + "\"\n";
          }
        
		line += "    ,\"description\": { \"author\": \"SystemML\" } \n" + "}" ;
        
        br.write(line);
        
        br.close(); 
        }catch (Exception e) {
			throw new IOException(e);
		}
	}
	
	
	public static void writeScalarMetaDataFile ( String mtdfile, ValueType v ) throws IOException {
		
        Path pt=new Path(mtdfile);
        FileSystem fs = FileSystem.get(_rJob);
        BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));		
		
        try {
          String line = "";
          line += "{ \n" +
                  "    \"" +  DataExpression.DATATYPEPARAM         +  "\": \"scalar\"\n" +
        		  "    ,\"" +  DataExpression.VALUETYPEPARAM        +  "\": ";
        		        
          switch (v) {
        	case DOUBLE:
        		line += "\"double\"\n";
        		break;
        	case INT:
        		line += "\"int\"\n";
        		break;
        	case BOOLEAN:
        		line += "\"boolean\"\n";
        		break;
        	case STRING:
        		line += "\"string\"\n";
        		break;
          };
          
          line += "    ,\"" + DataExpression.FORMAT_TYPE	+  "\": \"text\"\n" + 
                  "    ,\"description\": { \"author\": \"SystemML\" } \n" +" }" ;
        
        br.write(line);
        
        br.close();
        }catch (Exception e) {
			throw new IOException(e);
		}
	}
	
	public static double[][] readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, 
			int brlen, int bclen) 
	throws IOException
	{
		double[][] array=new double[(int)rlen][(int)clen];
	//	String filename = getSubDirsIgnoreLogs(dir);
		JobConf job = new JobConf();
		FileInputFormat.addInputPath(job, new Path(dir));
		
		try {

			InputFormat informat=inputinfo.inputFormatClass.newInstance();
			if(informat instanceof TextInputFormat)
				((TextInputFormat)informat).configure(job);
			InputSplit[] splits= informat.getSplits(job, 1);
			
			Converter inputConverter=MRJobConfiguration.getConverterClass(inputinfo, brlen, bclen, ConvertTarget.CELL).newInstance();
			inputConverter.setBlockSize(brlen, bclen);
    		
			Writable key=inputinfo.inputKeyClass.newInstance();
			Writable value=inputinfo.inputValueClass.newInstance();
			
			for(InputSplit split: splits)
			{
				RecordReader reader=informat.getRecordReader(split, job, Reporter.NULL);
				while(reader.next(key, value))
				{
					inputConverter.convert(key, value);
					while(inputConverter.hasNext())
					{
						Pair pair=inputConverter.next();
						MatrixIndexes index=(MatrixIndexes) pair.getKey();
						MatrixCell cell=(MatrixCell) pair.getValue();
						array[(int)index.getRowIndex()-1][(int)index.getColumnIndex()-1]=cell.getValue();
					}
				}
				reader.close();
			}
			
		} catch (Exception e) {
			throw new IOException(e);
		}
		return array;
	}
	
	public static double[] readColumnVectorFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen) 
	throws IOException
	{
		assert(clen==1 && bclen==1);
		double[][] array=readMatrixFromHDFS(dir, inputinfo, rlen, clen, brlen, bclen);
		double[] ret=new double[array.length];
		for(int i=0; i<array.length; i++)
			ret[i]=array[i][0];
		return ret;
	}
	
	public static double[] readRowVectorFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen) 
	throws IOException
	{
		assert(rlen==1 && brlen==1);
		double[][] array=readMatrixFromHDFS(dir, inputinfo, rlen, clen, brlen, bclen);
		return array[0];
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
		
		FSDataInputStream currentStream=fs.open(fileToRead);
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
			//System.out.println("**** numRead "+numRead+" -- "+readKey+": "+readValue + ", " + numRead + " " + cum_weight);
		}
	    
	    double ret = readKey.get();
	    if(average) {
	    	if(numRead<=offset+1) {
	    		reader.readNextKeyValuePairs(readKey, readValue);;
				numRead+=readValue.get();
				cum_weight += readValue.get();
				//System.out.println("**** numRead "+numRead+" -- "+readKey+": "+readValue + ", " + numRead + " " + cum_weight);
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
		//similarly setting dfs.datanode.data.dir.perm as no effect either.
	}
}
