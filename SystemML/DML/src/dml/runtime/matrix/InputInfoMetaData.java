package dml.runtime.matrix;

import dml.runtime.matrix.io.InputInfo;

public class InputInfoMetaData extends MatrixDimensionsMetaData {

	protected InputInfo info;
	
	InputInfoMetaData(MatrixCharacteristics mc, InputInfo info_ ) {
		super(mc);
		info = info_;
	}
	
	public InputInfo getInputInfo() {
		return info;
	}
}
