package com.ibm.bi.dml.runtime.transform;
import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.json.java.JSONObject;

public class ApplyTfCSVMapper implements Mapper<LongWritable, Text, NullWritable, Text> {
	
	ApplyTfHelper tfmapper = null;
	
	@Override
	public void configure(JobConf job) {
		try {
			tfmapper = new ApplyTfHelper(job);
			JSONObject spec = tfmapper.parseSpec();
			tfmapper.setupTfAgents(spec);
			tfmapper.loadTfMetadata(spec);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	@Override
	public void map(LongWritable rawKey, Text rawValue, OutputCollector<NullWritable, Text> out, Reporter reporter) throws IOException  {
		
		// output the header line
		if ( rawKey.get() == 0 && tfmapper._partFileWithHeader ) 
		{
			int numColumnsTf = tfmapper.processHeaderLine(rawValue);
			reporter.incrCounter(MRJobConfiguration.DataTransformCounters.TRANSFORMED_NUM_COLS, numColumnsTf);
			
			if ( tfmapper._hasHeader )
				return;
		}
		
		// parse the input line and apply transformation
		String[] words = tfmapper.getWords(rawValue);
		words = tfmapper.apply(words);
		
		try
		{
			String outStr = tfmapper.checkAndPrepOutputString(words);
			out.collect(NullWritable.get(), new Text(outStr));
		}
		catch(DMLRuntimeException e)
		{
			throw new IOException(e.getMessage() + ": " + rawValue.toString());
		}
	}

	@Override
	public void close() throws IOException {
	}

}
