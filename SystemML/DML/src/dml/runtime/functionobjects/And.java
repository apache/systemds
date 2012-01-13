package dml.runtime.functionobjects;

public class And extends ValueFunction {
	public static And singleObj = null;

	private And() {
		// nothing to do here
	}
	
	public static And getAndFnObject() {
		if ( singleObj == null )
			singleObj = new And();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public boolean execute(boolean in1, boolean in2) {
		return in1 && in2;
	}

}
