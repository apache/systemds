package dml.runtime.matrix.io;

import org.apache.hadoop.io.IntWritable;

public class TaggedInt extends Tagged<IntWritable>{

	public TaggedInt()
	{
		tag=-1;
		base=new IntWritable();
	}

	public TaggedInt(IntWritable b, byte t) {
		super(b, t);
	}
	
	public int hashCode()
	{
		return base.hashCode()+tag;
	}
	
	public int compareTo(TaggedInt other)
	{
		if(this.tag!=other.tag)
			return (this.tag-other.tag);
		else if(this.base.get()!=other.base.get())
			return (this.base.get()-other.base.get());
		return 0;
	}

	public boolean equals(TaggedInt other)
	{
	//	LOG.info("calling equals for MatrixCellIndexes!");
		return (this.tag==other.tag && this.base.get()==other.base.get());
	}
	
	public boolean equals(Object other)
	{
	//	LOG.info("calling equals for MatrixCellIndexes!");
		if( !(other instanceof TaggedInt))
			return false;
		return equals((TaggedInt)other);
	}
}
