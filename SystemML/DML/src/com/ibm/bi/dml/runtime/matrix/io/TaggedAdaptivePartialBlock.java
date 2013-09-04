package com.ibm.bi.dml.runtime.matrix.io;

public class TaggedAdaptivePartialBlock extends Tagged<AdaptivePartialBlock>
{
	public TaggedAdaptivePartialBlock(AdaptivePartialBlock b, byte t) 
	{
		super(b, t);
	}

	public TaggedAdaptivePartialBlock()
	{        
        tag = -1;
     	base = new AdaptivePartialBlock();
	}
}
