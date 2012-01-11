package dml.runtime.matrix.sort;

import java.io.DataOutputStream;
import java.io.IOException;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordWriter;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.Progressable;

public class CompactOutputFormat<K extends Writable, V extends Writable> extends FileOutputFormat<K,V> {

	static final String FINAL_SYNC_ATTRIBUTE = "final.sync";

	  /**
	   * Set the requirement for a final sync before the stream is closed.
	   */
	  public static void setFinalSync(JobConf conf, boolean newValue) {
	    conf.setBoolean(FINAL_SYNC_ATTRIBUTE, newValue);
	  }

	  /**
	   * Does the user want a final sync at close?
	   */
	  public static boolean getFinalSync(JobConf conf) {
	    return conf.getBoolean(FINAL_SYNC_ATTRIBUTE, false);
	  }
	
	public RecordWriter<K,V> getRecordWriter(FileSystem ignored, JobConf job, String name, Progressable progress) 
	throws IOException {
		
		Path file = FileOutputFormat.getTaskOutputPath(job, name);
		FileSystem fs = file.getFileSystem(job);
		FSDataOutputStream fileOut = fs.create(file, progress);
		return new FixedLengthRecordWriter<K,V>(fileOut, job);
	}
	
	public static class FixedLengthRecordWriter<K extends Writable, V extends Writable> implements RecordWriter<K, V> {

		private DataOutputStream out;
		 private boolean finalSync = false;
		
		public FixedLengthRecordWriter(DataOutputStream out, JobConf conf) {
			this.out = out;
			 finalSync = getFinalSync(conf);
		}
		
		@Override
		public void close(Reporter reporter) throws IOException {
			if (finalSync) {
		        ((FSDataOutputStream) out).sync();
		      }
			out.close();
		}

		@Override
		public void write(K key, V value) throws IOException {
			key.write(out);
			value.write(out);
		}	
		
	}
	
}
