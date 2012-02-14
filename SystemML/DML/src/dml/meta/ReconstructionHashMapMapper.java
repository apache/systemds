package dml.meta;

import java.io.IOException;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import dml.runtime.matrix.io.Converter;
import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.Pair;
import dml.runtime.matrix.mapred.MRJobConfiguration;
import dml.runtime.util.MapReduceTool;

public class ReconstructionHashMapMapper extends MapReduceBase
implements Mapper<Writable, Writable, LongWritable, ReconstructionHashMapMapOutputValue> {
	
	private Converter inputConverter=null;
	PartitionParams pp = new PartitionParams() ;
	int brlen, bclen ;
	MultipleOutputs multipleOutputs ;

	HashMap <Long, Long[]> thehashmap;
	int thisfold;
	
	@Override
	public void map(Writable rawKey, Writable rawValue,
			OutputCollector<LongWritable, ReconstructionHashMapMapOutputValue> out, Reporter reporter)
	throws IOException {
		inputConverter.setBlockSize(brlen, bclen);
		inputConverter.convert(rawKey, rawValue);
		while(inputConverter.hasNext()) {
			Pair<MatrixIndexes, MatrixBlock> pair=inputConverter.next();
			//bmm.execute(pair, reporter, out) ;
			//carry out the reconstruction; the map output key is the blknumber this subrow goes out to;
			//the map output value has <subrowid within that output blk, actual entry in that cell)>
			long blky = pair.getKey().getRowIndex() - 1;	//systemml matrixblks start from (1,1)
			int rpb = pp.rows_in_block;		//TODO: assuming this is the general block y dimension, or numrows
			long N = thehashmap.size();	//the number of cols in orig matrix, each has a tuple
			int nrows = pair.getValue().getNumRows();
			//go thro hashmap & obtain map from  absolrowid of this matrix's rows to orig tuplekeys (cols) for this fold
			long[] reversemap = new long[nrows];
			long numtrain = 0, index = 0;
			while ((index < N) && (numtrain < blky * rpb + nrows)) {
				long code = thehashmap.get(index)[thisfold];
				if(code < 0)	//count only train entries
					numtrain++;
				if (numtrain > blky * rpb)
					reversemap[(int)(numtrain - blky * rpb - 1)] = index;	//the cast wont cause overflows
				index++;
			}
			//we need reconstruction only for cv holdout and el rsm - 
			//TODO TODO: reducer reconstructs blks of final coeff matr!
			//TODO TODO: similarly, join based method!! that involves writing out to 1x1 blks and Reblock afterwards!			
			for(int r=0; r < nrows; r++) {
				int futsubrowid = (int) (reversemap[r] % rpb);		//subrowsid in future matrix blk; cast shld work
				long futblk = reversemap[r] / rpb;					//the rounding off shld give correct blk num
				double thisentry = pair.getValue().getValue(r,0);	//single column matrix
				LongWritable outkey = new LongWritable(futblk);
				ReconstructionHashMapMapOutputValue outval = new ReconstructionHashMapMapOutputValue(
																		futsubrowid, thisentry);
				out.collect(outkey, outval);
			}			
		}
	}
	
	public void close() throws IOException  {
		multipleOutputs.close();
	}
	
	@Override
	public void configure(JobConf job) {
		multipleOutputs = new MultipleOutputs(job) ;
		//get input converter information
		inputConverter=MRJobConfiguration.getInputConverter(job, (byte)0);
		brlen=MRJobConfiguration.getNumRowsPerBlock(job, (byte)0);
		bclen=MRJobConfiguration.getNumColumnsPerBlock(job, (byte)0);		
		pp = MRJobConfiguration.getPartitionParams(job) ;

		thisfold = job.getInt("foldnum", 0);	//get the fold num corresp to this train o/p col matrx
		thehashmap = new HashMap<Long, Long[]>();
		//read the hashmap into memory
		try {
			readHashMap(job);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InstantiationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void readHashMap(JobConf job) throws IOException, IllegalAccessException, InstantiationException {
		Configuration sfconf = new Configuration();
		sfconf.set("CompressionType", "BLOCK");
		FileSystem fs = FileSystem.get(sfconf);
		/*FileSystem fs = FileSystem.getLocal(sfconf);
		Path[] localFiles = DistributedCache.getLocalCacheFiles(job);
		if(localFiles == null) {	//TODO -> why is the distr cache not working?! localfiles is null!
			System.out.println("$$$$$$$$$$$$$$$\t localfiles is null! \t$$$$$$$$$$$$$$$$$$$\n");
		}
		else {
			System.out.println("$$$$$$$$$$$$$$$\t localfiles[0] is " + localFiles[0] + "! \t$$$$$$$$$$$$$$$$$$$\n");
		}
		Reader sfreader = new SequenceFile.Reader(fs, localFiles[0], sfconf);
		*/
		//TODO: distr cache doesnt work! localfiles is null
		Reader sfreader = new SequenceFile.Reader(fs, new Path(pp.sfmapfile), sfconf);
		WritableComparable key = (WritableComparable) sfreader.getKeyClass().newInstance();
		Writable value = (Writable) sfreader.getValueClass().newInstance();
		while(sfreader.next(key, value)) {
			System.out.println ("read in a key-value pair! key: " + key.toString() + " and value: " + value.toString());
			thehashmap.put(((LongWritable)key).get(), ((WritableLongArray)value).array);
		}
		sfreader.close();
		System.out.println("$$$$$$$$\t Read in the hashmap into memory! \t $$$$$$$$\n");
	}
	
	
}
