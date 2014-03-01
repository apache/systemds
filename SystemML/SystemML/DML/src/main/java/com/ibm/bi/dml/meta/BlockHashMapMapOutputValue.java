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

import org.apache.hadoop.io.Writable;


public class BlockHashMapMapOutputValue implements Writable 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	int auxdata;
	//MatrixBlock blk;
	
	//instead of sending out subblks with auxdata indicating posn, i send out the cell value, with another position locator
	//TODO this menas changes in blkhshmapmaputval, btstrpmpr, holdoutmpr, kfoldmpr, and reducer!!
	int locator; //if we send subrowblks, this gives colid within that future blk
	double cellvalue;
	
	public BlockHashMapMapOutputValue() {
		auxdata = 0;
		//blk = new MatrixBlock();
		cellvalue = locator = 0; 
	}
	
	public BlockHashMapMapOutputValue(BlockHashMapMapOutputValue that) {
		auxdata = that.auxdata;
		//blk = new MatrixBlock();
		//blk = that.blk;
		cellvalue = that.cellvalue;
		locator = that.locator;
	}
	@Override
	public void readFields(DataInput inp) throws IOException {
		auxdata = inp.readInt();
		//blk.readFields(inp);
		cellvalue = inp.readDouble();
		locator = inp.readInt();
	}
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(auxdata);
		//blk.write(out);
		out.writeDouble(cellvalue);
		out.writeInt(locator);
	}
	 public void print() {
		 System.out.println(toString());
		 //blk.print();
	 }
		
	 public String toString() {;
	 	return "auxdata: " + auxdata + "\n" 
	 		//+ blk.toString();
	 		+ "locator: " + locator + " cellval: " + cellvalue + "\n";
	 }

}
//</Arun>