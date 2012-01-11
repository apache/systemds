package dml.runtime.matrix.io;

public class TaggedMatrixBlock extends TaggedMatrixValue{
	
	public TaggedMatrixBlock(MatrixBlock b, byte t) {
		super(b, t);
	}

	public TaggedMatrixBlock()
	{        
        tag=-1;
     	base=new MatrixBlock();
	}

	public TaggedMatrixBlock(TaggedMatrixBlock that) {
		tag=that.getTag();
		base=new MatrixBlock();
		base.copy(that.getBaseObject());
	}
}