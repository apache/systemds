package gnmf.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.RawComparator;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;

public class TaggedIndex implements WritableComparable<TaggedIndex>
{
	public static final int TYPE_MATRIX		= 0;
	public static final int TYPE_VECTOR		= 1;
	public static final int TYPE_CELL		= 2;
	
	public static final int TYPE_VECTOR_HW	= 1;
	public static final int TYPE_VECTOR_X	= 2;
	public static final int TYPE_VECTOR_Y	= 3;
	
	
	private long index;
	private int type;
	
	
	public TaggedIndex()
	{
		
	}
	
	public TaggedIndex(long index, int type)
	{
		this.index = index;
		this.type = type;
	}
	
	public long getIndex()
	{
		return index;
	}
	
	public int getType()
	{
		return type;
	}

	@Override
	public void readFields(DataInput in) throws IOException
	{
		index = in.readLong();
		type = in.readByte();
	}

	@Override
	public void write(DataOutput out) throws IOException
	{
		out.writeLong(index);
		out.writeByte(type);
	}
	
	@Override
	public int compareTo(TaggedIndex that)
	{
		if(this.index != that.index)
			return (this.index > that.index ? 1 : -1);
		else if(this.type != that.type)
			return (this.type - that.type);
		else
			return 0;
	}
	
	public int hashCode()
	{
		return Long.valueOf(index).hashCode();
	}
	
	public static class Comparator implements RawComparator<TaggedIndex>
	{
		@Override
		public int compare(byte[] b1, int s1, int l1, byte[] b2,
				int s2, int l2)
		{
			return WritableComparator.compareBytes(b1, s1, l1, b2, s2, l2);
		}

		@Override
		public int compare(TaggedIndex i1, TaggedIndex i2)
		{
			return i1.compareTo(i2);
		}
	}
	
	public static class SecondarySort implements RawComparator<TaggedIndex>
	{
		@Override
		public int compare(byte[] b1, int s1, int l1, byte[] b2,
				int s2, int l2)
		{
			return WritableComparator.compareBytes(b1, s1, l1, b2, s2, l2);
		}

		@Override
		public int compare(TaggedIndex i1, TaggedIndex i2)
		{
			if(i1.getIndex() != i2.getIndex())
				return (i1.getIndex() > i2.getIndex() ? 1 : -1);
			if(i1.getType() != i2.getType())
				return (i1.getType() - i2.getType());
			else
				return 0;
		}
	}
	
	public static class IndexPartitioner implements Partitioner<TaggedIndex, MatrixObject>
	{
		@Override
		public int getPartition(TaggedIndex key, MatrixObject value,
				int numPartitions)
		{
			return (int) key.getIndex() % numPartitions;
		}

		@Override
		public void configure(JobConf job)
		{
			
		}
	}
}
