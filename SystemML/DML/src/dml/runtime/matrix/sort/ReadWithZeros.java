package dml.runtime.matrix.sort;

import java.io.IOException;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;

public class ReadWithZeros {

	private boolean contain0s=false;
	private long numZeros=0;
	private FSDataInputStream currentStream;
	
	private DoubleWritable keyAfterZero=new DoubleWritable();
	private IntWritable valueAfterZero=new IntWritable(); 
	private boolean justFound0=false;
	
	public ReadWithZeros(FSDataInputStream in, boolean contain0, long num0)
	{
		currentStream=in;
		contain0s=contain0;
		numZeros=num0;
	}
	
	public void readNextKeyValuePairs(DoubleWritable readKey, IntWritable readValue)throws IOException 
	{
		if(contain0s && justFound0)
		{
			readKey.set(keyAfterZero.get());
			readValue.set(valueAfterZero.get());
			contain0s=false;
		}else
		{
			readKey.readFields(currentStream);
			readValue.readFields(currentStream);
		}
		
		if(contain0s && !justFound0 && readKey.get()>=0)
		{
			justFound0=true;
			keyAfterZero.set(readKey.get());
			valueAfterZero.set(readValue.get());
			readKey.set(0);
			readValue.set((int)numZeros);
		}
	}
}
