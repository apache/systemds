package dml.meta;
//<Arun>
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.Pair;

public abstract class BlockHashMapMapperMethod {
	MatrixIndexes mi = new MatrixIndexes() ;
	PartitionParams pp ;
	MultipleOutputs multipleOutputs ;

	//HashMap <String, Integer[]> thehashmap;
	//HashMap <Long, Long[]> thehashmap;
	VectorOfArraysBag thehashmap;		//im now using an array-based hashmap since otherwise HM had poorer perf than JR for large m!! (HM-cell-array)
									//here, i store 2^30 longs in each [] (the signed int limit), and concat them logically in a vector
									//an update on this - for btstrp, we store aux locn indicators, bringisnz size from MT to (2+f)MT
	public BlockHashMapMapperMethod () {
		mi = null;
		pp = null;
		//thehashmap = new HashMap <Long, Long[]>() ;
		thehashmap = null;
	}
	
	public BlockHashMapMapperMethod(PartitionParams pp, MultipleOutputs multipleOutputs) {
		this.pp = pp ;
		this.multipleOutputs = multipleOutputs ;
		//thehashmap = new HashMap <Long, Long[]>() ;
		thehashmap = null;
	}
	
	abstract void execute(Pair<MatrixIndexes, MatrixBlock> pair, Reporter reporter, OutputCollector out) 
	throws IOException ;
	
	public void sfReadHashMap(JobConf job, long length, int width) throws IOException, IllegalAccessException, InstantiationException {
		//thehashmap = new VectorOfArrays(length, width);
		if(pp.isEL == false && pp.cvt == PartitionParams.CrossvalType.bootstrap || 
				pp.isEL == true && pp.et == PartitionParams.EnsembleType.bagging)
			thehashmap = new VectorOfArraysBag(length, width, pp.frac);	//(2+2f)MT here
		else
			thehashmap = new VectorOfArraysBag(length, width, 0);		//2MT here (can be made bk into MT, but ignored for now)
		
		Path[] localFiles = DistributedCache.getLocalCacheFiles(job);
		//System.out.println("In sfreadhashmap, job(mapred.cachce.files) is " + job.get("mapred.cache.files"));		
		FileSystem fs = FileSystem.getLocal(job);
		if(localFiles == null) {
			//System.out.println("$$$$$$$$$$$$$$$\t localfiles is null! \t$$$$$$$$$$$$$$$$$$$\n");
			localFiles = new Path[]{new Path(job.get("mapred.cache.files"))};	//we know only one file is set in distr cache! TODO
		}
		Configuration sfconf = new Configuration();
		sfconf.set("CompressionType", "BLOCK");
		Reader sfreader = new SequenceFile.Reader(fs, localFiles[0], sfconf);
			WritableComparable key = (WritableComparable) sfreader.getKeyClass().newInstance();
		Writable value = (Writable) sfreader.getValueClass().newInstance();
		while(sfreader.next(key, value)) {
			//System.out.println ("read in a key-value pair! key: " + key.toString() + " and value: " + value.toString());
			//thehashmap.put(((LongWritable)key).get(), ((WritableLongArray)value).array);
			WritableLongArray theval = (WritableLongArray)value;
			for(int i=0; i<theval.length; i++) {
				if(pp.isEL == false && pp.cvt == PartitionParams.CrossvalType.bootstrap || 
						pp.isEL == true && pp.et == PartitionParams.EnsembleType.bagging)
					thehashmap.set(theval.array[i], i, ((LongWritable)key).get());	//switch entryid and entry (prevrowid to futrowid now)
				else
					thehashmap.set(((LongWritable)key).get(), i, theval.array[i]);
			}
		}
		sfreader.close();
		System.out.println("$$$$$$$$ Read in the hashmap into memory! $$$$$$$$");
	}
	
	//i dont use these anymore since ive moved HM to cell based kv pairs due to 1/c vs sparsity
	
}
//</Arun>