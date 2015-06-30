package com.ibm.bi.dml.runtime.transform;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ByteWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.instructions.mr.CSVReblockInstruction;
import com.ibm.bi.dml.runtime.matrix.CSVReblockMR;
import com.ibm.bi.dml.runtime.matrix.CSVReblockMR.OffsetCount;
import com.ibm.bi.dml.runtime.matrix.data.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.CSVReblockMapper;
import com.ibm.bi.dml.runtime.matrix.mapred.CSVReblockMapper.IndexedBlockRow;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.matrix.mapred.MapperBase;
import com.ibm.json.java.JSONObject;

@SuppressWarnings("deprecation")
public class ApplyTfBBMapper extends MapperBase implements Mapper<LongWritable, Text, TaggedFirstSecondIndexes, CSVReblockMR.BlockRow>{
	
	ApplyTfHelper tfmapper = null;
	
	// variables relevant to CSV Reblock
	private IndexedBlockRow idxRow = null;
	private long rowOffset=0;
	private HashMap<Long, Long> offsetMap=new HashMap<Long, Long>();
	private boolean _first = true;
	private long num=0;
	
	@Override
	public void configure(JobConf job) {
		super.configure(job);
		try {
			tfmapper = new ApplyTfHelper(job);
			JSONObject spec = tfmapper.parseSpec();
			tfmapper.setupTfAgents(spec);
			tfmapper.loadTfMetadata(spec);
			
			// Load relevant information for CSV Reblock
			
			ByteWritable key=new ByteWritable();
			OffsetCount value=new OffsetCount();
			Path p=new Path(job.get(CSVReblockMR.ROWID_FILE_NAME));
			
			FileSystem fs = FileSystem.get(job);
			Path thisPath=new Path(job.get("map.input.file")).makeQualified(fs);
			String thisfile=thisPath.toString();

			SequenceFile.Reader reader = new SequenceFile.Reader(fs, p, job);
			while (reader.next(key, value)) {
				// "key" needn't be checked since the offset file has information about a single CSV input (the raw data file)
				if(thisfile.equals(value.filename))
					offsetMap.put(value.fileOffset, value.count);
			}
			reader.close();

			idxRow = new CSVReblockMapper.IndexedBlockRow();
			int maxBclen=0;
		
			for(ArrayList<CSVReblockInstruction> insv: csv_reblock_instructions)
				for(CSVReblockInstruction in: insv)
				{	
					if(maxBclen<in.bclen)
						maxBclen=in.bclen;
				}
			
			//always dense since common csv usecase
			idxRow.getRow().data.reset(1, maxBclen, false);		

		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	@Override
	public void map(LongWritable rawKey, Text rawValue, OutputCollector<TaggedFirstSecondIndexes,CSVReblockMR.BlockRow> out, Reporter reporter) throws IOException  {
		
		if(_first) {
			rowOffset=offsetMap.get(rawKey.get());
			_first=false;
		}
		
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

		try {
			tfmapper.check(words);
		}
		catch(DMLRuntimeException e)
		{
			throw new RuntimeException(e.getMessage() + ":" + rawValue.toString());
		}
		
		// Perform CSV Reblock
		
		CSVReblockInstruction ins = csv_reblock_instructions.get(0).get(0);
		try {
			idxRow = CSVReblockMapper.processRow(idxRow, words, rowOffset, num, ins.output, ins.brlen, ins.bclen, ins.fill, ins.fillValue, out);
		} catch(Exception e) {
			throw new RuntimeException(e);
		}
		num++;
		
		//out.collect(NullWritable.get(), new Text(sb.toString()));
	}

	@Override
	public void close() throws IOException {
	}

	@Override
	protected void specialOperationsForActualMap(int index,
			OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException {
	}

}
