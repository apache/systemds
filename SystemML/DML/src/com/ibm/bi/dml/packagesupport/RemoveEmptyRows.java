package com.ibm.bi.dml.packagesupport;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.StringTokenizer;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.nimble.hadoop.HDFSFileManager;

import com.ibm.bi.dml.packagesupport.FIO;
import com.ibm.bi.dml.packagesupport.Matrix;
import com.ibm.bi.dml.packagesupport.PackageFunction;
import com.ibm.bi.dml.packagesupport.Matrix.ValueType;
//import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;

/**
 * 
 *
 */
public class RemoveEmptyRows extends PackageFunction 
{	
	private static final long serialVersionUID = 1L;
	private final String OUTPUT_FILE = "TMP";
	
	private Matrix _ret; 
	
	
	@Override
	public int getNumFunctionOutputs() 
	{
		return 1;
	}

	@Override
	public FIO getFunctionOutput(int pos) 
	{
		return _ret;
	}

	@Override
	public void execute() 
	{
		//Timing time = new Timing();
		//time.start();
		
		Matrix mat = (Matrix) this.getFunctionInput(0);
		String fnameOld = mat.getFilePath(); 
		
		HashMap<Long,Long> keyMap = new HashMap<Long,Long>(); //old,new rowID
			
		try
		{		
			//prepare input
			JobConf job = new JobConf();	
			Path path = new Path(fnameOld);
			FileSystem fs = FileSystem.get(job);
			if( !fs.exists(path) )	
				throw new IOException("File "+fnameOld+" does not exist on HDFS.");
			FileInputFormat.addInputPath(job, path); 
			TextInputFormat informat = new TextInputFormat();
			informat.configure(job);
			
			//prepare output
			String fnameNew = createOutputFilePathAndName( OUTPUT_FILE );
			DataOutputStream ostream = HDFSFileManager.getOutputStreamStatic( fnameNew, true);
		
			//read and write if necessary
			InputSplit[] splits = informat.getSplits(job, 1);
		
			LongWritable key = new LongWritable();
			Text value = new Text();
			long ID = 1;
			
			try
			{
				for(InputSplit split: splits)
				{
					RecordReader<LongWritable,Text> reader = informat.getRecordReader(split, job, Reporter.NULL);				
					try
					{
						while( reader.next(key, value) )
						{
							String cellStr = value.toString().trim();							
							StringTokenizer st = new StringTokenizer(cellStr, " ");
							long row = Integer.parseInt( st.nextToken() );
							long col = Integer.parseInt( st.nextToken() );
							double lvalue = Double.parseDouble( st.nextToken() );
							
							if( !keyMap.containsKey( row ) )
								keyMap.put(row, ID++);
							long rowNew = keyMap.get( row );
							
							StringBuilder sb = new StringBuilder();
							sb.append(rowNew);
							sb.append(" ");
							sb.append(col);
							sb.append(" ");
							sb.append(lvalue);
							sb.append("\n");
							
							ostream.writeBytes( sb.toString() );	
						}
					}
					finally
					{
						if( reader != null )
							reader.close();
					}
				}
				
				_ret = new Matrix(fnameNew, keyMap.size(), mat.getNumCols(), ValueType.Double);
			}
			finally
			{
				if( ostream != null )
					ostream.close();	
			}
		}
		catch(Exception ex)
		{
			throw new RuntimeException( "Unable to execute external function.", ex );
		}
		
		//System.out.println("Executed external function in "+time.stop()+"ms");
	}
}
