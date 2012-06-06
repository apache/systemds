package com.ibm.bi.dml.runtime.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
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

//import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDHandler; TODO
import com.ibm.bi.dml.parser.Statement;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.Converter;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.NumItemsByEachReducerMetaData;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.Pair;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.matrix.sort.ReadWithZeros;


public class MapReduceTool {
	// private static final Log LOG = LogFactory.getLog(AggregateReducer.class);
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

	
	public static int getUniqueMapperId(JobConf job, boolean inMapper) {
		//TODO: investigate ID pattern, required for parallel jobs
		/*String nodePrefix = job.get("mapred.task.id"); 
		return IDHandler.extractIntID(nodePrefix);*/
		
		String nodePrefix = job.get("mapred.task.id");
		int j = nodePrefix.lastIndexOf("_");
		nodePrefix = nodePrefix.substring(j + 1);
		// System.out.println("nodePrefix = " + nodePrefix) ;
		return (new Integer(nodePrefix)).intValue();
	}

	public static String getGloballyUniqueName(JobConf job) {
		return job.get("mapred.task.id");
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
		JobConf job = new JobConf();
		Path outpath = new Path(dir);
		if (FileSystem.get(job).exists(outpath)) {
			//System.err.println("Deleting " + outpath + " ... ");
			FileSystem.get(job).delete(outpath, true);
			//if ( ! FileSystem.get(job).delete(outpath, true) )
				//System.err.println("delete failed: " + outpath );
/*			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
*/			//System.out.println("deleted..");
		}
	}

	public static boolean isHDFSDirectory(String dir) throws IOException {
		FileSystem fs = FileSystem.get(new JobConf());
		Path pth = new Path(dir);
		FileStatus fstat = fs.getFileStatus(pth);
		return fstat.isDir();
	}

	public static boolean isHDFSFileEmpty(String dir) throws IOException {
		FileSystem fs = FileSystem.get(new JobConf());
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
		JobConf job = new JobConf();
		Path originalpath = new Path(originalDir);
		
		deleteFileIfExistOnHDFS(newDir);
		Path newpath = new Path(newDir);
		
		if (FileSystem.get(job).exists(originalpath)) {
			FileSystem fs = FileSystem.get(job);
			fs.rename(originalpath, newpath);
		}
	}

	public static void copyFileOnHDFS(String originalDir, String newDir) throws IOException {
		JobConf job = new JobConf();
		Path originalpath = new Path(originalDir);
		Path newpath = new Path(newDir);
		if (FileSystem.get(job).exists(originalpath)) {
			FileSystem fs = FileSystem.get(job);
			fs.rename(originalpath, newpath);
		}
	}

	public static String getSubDirs(String dir) throws IOException {
		FileSystem fs = FileSystem.get(new Configuration());
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
		FileSystem fs = FileSystem.get(new Configuration());
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
	
	private static BufferedReader setupScalarInputFile ( String filename ) throws IOException {
        Path pt=new Path(filename);
        FileSystem fs = FileSystem.get(new Configuration());
        BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(pt)));		
        return br;
	}
	
	public static double readDoubleFromHDFSFile(String filename) throws IOException {
		BufferedReader br = setupScalarInputFile(filename);
		String line = br.readLine();
		br.close();
		return Double.parseDouble(line);
	}
	public static int readIntegerFromHDFSFile(String filename) throws IOException {
		BufferedReader br = setupScalarInputFile(filename);
		String line = br.readLine();
		br.close();
		return Integer.parseInt(line);
	}
	public static boolean readBooleanFromHDFSFile(String filename) throws IOException {
		BufferedReader br = setupScalarInputFile(filename);
		String line = br.readLine();
		br.close();
		return Boolean.parseBoolean(line);
	}
	public static String readStringFromHDFSFile(String filename) throws IOException {
		BufferedReader br = setupScalarInputFile(filename);
		// handle multi-line strings in the HDFS file
		String output = "", temp = "";
		output = br.readLine();
		while ( (temp = br.readLine()) != null ) {
			output += "\n" + temp;
		}
		br.close();
		return output;
	}
		
	private static BufferedWriter setupScalarOutputFile ( String filename ) throws IOException {
        Path pt=new Path(filename);
        FileSystem fs = FileSystem.get(new Configuration());
        BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));		
        return br;
	}
	
	public static void writeDoubleToHDFS ( double d, String filename ) throws IOException {
        BufferedWriter br = setupScalarOutputFile(filename);
        String line = "" + d;
        br.write(line);
        br.close();
	}
	
	public static void writeIntToHDFS ( int i, String filename ) throws IOException {
        BufferedWriter br = setupScalarOutputFile(filename);
        String line = "" + i;
        br.write(line);
        br.close();
	}
	
	public static void writeBooleanToHDFS ( boolean b, String filename ) throws IOException {
        BufferedWriter br = setupScalarOutputFile(filename);
        String line = "" + b;
        br.write(line);
        br.close();
	}
	
	public static void writeStringToHDFS ( String s, String filename ) throws IOException {
        BufferedWriter br = setupScalarOutputFile(filename);
        String line = "" + s;
        br.write(line);
        br.close();
	}
	
	public static void writeMetaDataFile ( String mtdfile, MatrixCharacteristics mc, OutputInfo outinfo ) throws IOException {
		//MatrixCharacteristics mc = ((MatrixDimensionsMetaData) md).getMatrixCharacteristics();
        Path pt=new Path(mtdfile);
        FileSystem fs = FileSystem.get(new Configuration());
        BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));		
		
        String line = "";
        line += "{ \n" +
        "    \"" +  Statement.READROWPARAM 			+  "\": " + mc.numRows + "\n" + 
		"    ,\"" + Statement.READCOLPARAM 			+  "\": " + mc.numColumns + "\n" + 
		"    ,\"" + Statement.ROWBLOCKCOUNTPARAM	+  "\": " + mc.numRowsPerBlock + "\n" + 
		"    ,\"" + Statement.COLUMNBLOCKCOUNTPARAM +  "\": " + mc.numColumnsPerBlock + "\n" + 
		"    ,\"" +	Statement.READNUMNONZEROPARAM	+  "\": " + mc.nonZeros + "\n" +
		"    ,\"" + Statement.FORMAT_TYPE	+  "\": "; 
        if ( outinfo == OutputInfo.TextCellOutputInfo ) {
        	line += "\"text\"\n";
        } else if (outinfo == OutputInfo.BinaryBlockOutputInfo || outinfo == OutputInfo.BinaryCellOutputInfo ) {
        	line += "\"binary\"\n"; // currently, there is no way to differentiate between them
        } else {
        	line += "\"specialized\"\n"; // this should not be the final output info
        }
		line += "    ,\"description\": { \"author\": \"SystemML\" } \n" + 
		"}" ;
        
        br.write(line);
        
        br.close();
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
			
			Converter inputConverter=MRJobConfiguration.getConverterClass(inputinfo, false, brlen, bclen).newInstance();
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
	
	public static double pickValue(String dir, NumItemsByEachReducerMetaData metadata, double p) 
	throws IOException
	{
		long[] counts=metadata.getNumItemsArray();
		long[] ranges=new long[counts.length];
		ranges[0]=counts[0];
		for(int i=1; i<counts.length; i++)
			ranges[i]=ranges[i-1]+counts[i];
		
		long total=ranges[ranges.length-1];
		
		int currentPart=0;
		long pos=(long)Math.ceil(total*p);
		while(ranges[currentPart]<pos)
			currentPart++;
		int offset;
		if(currentPart>0)
			offset=(int)(pos-ranges[currentPart-1]-1);
		else
			offset=(int)pos-1;
		
		Configuration conf = new Configuration();
		FileSystem fs=FileSystem.get(conf);
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
			//System.out.println("**** numRead "+numRead+" -- "+readKey+": "+readValue);
			numRead+=readValue.get();
		}
	    currentStream.close();
		return readKey.get();
		
	}
	
	public static int extractNumberFromOutputFile(String name)
	{
		int i=name.indexOf("part-");
		assert(i>=0);
		return Integer.parseInt(name.substring(i+5));
	}
}
