/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

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
	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	ApplyTfHelper tfmapper = null;
	Reporter _reporter = null;
	
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
			_reporter = reporter;
			tfmapper.processHeaderLine(rawValue);
			if ( tfmapper._hasHeader )
				return;
		}
		
		// parse the input line and apply transformation
		String[] words = tfmapper.getWords(rawValue);
		if(!tfmapper.omit(words))
		{
			try {
				words = tfmapper.apply(words);
				String outStr = tfmapper.checkAndPrepOutputString(words);
				out.collect(NullWritable.get(), new Text(outStr));
			} catch(DMLRuntimeException e)
			{
				throw new RuntimeException(e.getMessage() + ": " + rawValue.toString());
			}
		}
	}

	@Override
	public void close() throws IOException {
		_reporter.incrCounter(MRJobConfiguration.DataTransformCounters.TRANSFORMED_NUM_ROWS, tfmapper.getNumTransformedRows());
		_reporter.incrCounter(MRJobConfiguration.DataTransformCounters.TRANSFORMED_NUM_COLS, tfmapper.getNumTransformedColumns());
	}

}
