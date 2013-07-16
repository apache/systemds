package com.ibm.bi.dml.runtime.matrix;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.matrix.io.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.MapperBase;

public class WriteCSVMR {

	public static class RowBlock implements Writable
	{
		public int numCols=0;
		public double[] container=null;
		
		@Override
		public void readFields(DataInput in) throws IOException {
			numCols=in.readInt();
			if(container==null || container.length<numCols)
				container=new double[numCols];
			for(int i=0; i<numCols; i++)
				container[i]=in.readDouble();
		}
		@Override
		public void write(DataOutput out) throws IOException {
			out.writeInt(numCols);
			for(int i=0; i<numCols; i++)
				out.writeDouble(container[i]);
		}
		
		public String toString()
		{
			String str="";
			for(int i=0; i<numCols; i++)
				str+=container[i]+", ";
			return str;
		}
		
	}	
	
	static class BreakMapper extends MapperBase implements Mapper<WritableComparable, Writable, TaggedFirstSecondIndexes, RowBlock>
	{

		@Override
		protected void specialOperationsForActualMap(int index,
				OutputCollector<Writable, Writable> out, Reporter reporter)
				throws IOException {
			// TODO Auto-generated method stub
			
		}

		@Override
		public void map(WritableComparable arg0, Writable arg1,
				OutputCollector<TaggedFirstSecondIndexes, RowBlock> arg2,
				Reporter arg3) throws IOException {
			// TODO Auto-generated method stub
			
		}
		
	}
}
