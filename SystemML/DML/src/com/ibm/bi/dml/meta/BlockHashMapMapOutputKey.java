/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.meta;
//<Arun>
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;

import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.util.UtilFunctions;


public class BlockHashMapMapOutputKey implements WritableComparable<BlockHashMapMapOutputKey> 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public int foldid;		//2i, 2i+1, i, etc
	public long blkx, blky;	//indexes of blk in future matrix
	
	public BlockHashMapMapOutputKey() {
		blkx = blky = foldid = -1;
	}
	public void WritableComparableHashMapKey(int f, long x, long y) {
		foldid = f;
		blkx = x;
		blky = y;
	}
	@Override
	public void readFields(DataInput inp) throws IOException {
		foldid = inp.readInt();
		blkx = inp.readLong();
		blky = inp.readLong();
	}
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(foldid);
		out.writeLong(blkx);
		out.writeLong(blky);
	}
	@Override
	public int compareTo(BlockHashMapMapOutputKey o) {
		if (this.foldid != o.foldid)
			return (this.foldid > o.foldid ? 1 : -1);
		else if (this.blkx != o.blkx)
			return (this.blkx > o.blkx ? 1 : -1);
		else if (this.blky != o.blky)
			return (this.blky > o.blky ? 1 : -1);
		return 0;
	}
	public boolean equals(BlockHashMapMapOutputKey other)
	{
		return (this.foldid==other.foldid && 
				this.blkx==other.blkx && 
				this.blky==other.blky);
	}
	public boolean equals(Object other) {
		if(!(other instanceof BlockHashMapMapOutputKey))
			return false;
		return (this.foldid==((BlockHashMapMapOutputKey)other).foldid && 
				this.blkx==((BlockHashMapMapOutputKey)other).blkx && 
				this.blky==((BlockHashMapMapOutputKey)other).blky);
	}
	public int hashCode() {
		return UtilFunctions.longHashFunc((blkx << 32)+(blky << 16)+foldid+MatrixIndexes.ADD_PRIME1)%MatrixIndexes.DIVIDE_PRIME;//TODO: check if ok!
		//return UtilFunctions.longHashFunc((long)foldid ^ blkx ^ blky);		//TODO: check if this is okay!
		//return UtilFunctions.longHashFunc((row<<32)+column+ADD_PRIME1)%DIVIDE_PRIME;
	}
	public void print() {
		System.out.println(toString());
	}		
	public String toString() {
		return "<" + foldid + ", (" + blkx + "," + blky + ")>";
	}
}
//</Arun>