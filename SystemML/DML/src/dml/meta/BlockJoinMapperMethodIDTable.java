package dml.meta;
//<Arun>
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import umontreal.iro.lecuyer.rng.WELL1024;

import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixCell;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.Pair;
import dml.runtime.matrix.io.PartialBlock;

public abstract class BlockJoinMapperMethodIDTable {
	MatrixIndexes mi = new MatrixIndexes() ;
	PartitionParams pp ;
	MultipleOutputs multipleOutputs ;
	
	public BlockJoinMapperMethodIDTable () {
		mi = null;
		pp = null;
	}
	
	public BlockJoinMapperMethodIDTable(PartitionParams pp, MultipleOutputs multipleOutputs) {
		this.pp = pp ;
		this.multipleOutputs = multipleOutputs ;
	}
	
	abstract void execute(LongWritable key, WritableLongArray value, Reporter reporter, OutputCollector out) 
	throws IOException ;
		
	public MatrixBlock getSubRowBlock(MatrixBlock blk, int rownum) {
		int ncols = blk.getNumColumns();
		MatrixBlock thissubrowblk = new MatrixBlock(1, ncols, true);	//presume sparse
		//populate subrowblock
		for(int c=0; c<ncols; c++) {
			thissubrowblk.setValue(rownum, c, blk.getValue(rownum, c));
		}
		thissubrowblk.examSparsity();	//refactor based on sparsity
		return thissubrowblk;
	}
}
//</Arun>