package dml.runtime.functionobjects;

public class Or extends ValueFunction {
	public static Or singleObj = null;

	private Or() {
		// nothing to do here
	}
	
	public static Or getOrFnObject() {
		if ( singleObj == null )
			singleObj = new Or();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public boolean execute(boolean in1, boolean in2) {
		return in1 || in2;
	}

}
