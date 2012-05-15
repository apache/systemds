package com.ibm.bi.dml.runtime.matrix.io;

public class TaggedPartialBlock extends Tagged<PartialBlock>{
	public TaggedPartialBlock(PartialBlock b, byte t) {
		super(b, t);
	}

	public TaggedPartialBlock()
	{        
        tag=-1;
     	base=new PartialBlock();
	}
}
