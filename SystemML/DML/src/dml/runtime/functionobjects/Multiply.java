package dml.runtime.functionobjects;

public class Multiply extends ValueFunction {

	private static Multiply singleObj = null;
	
	private Multiply() {
		// nothing to do here
	}
	
	public static Multiply getMultiplyFnObject() {
		if ( singleObj == null )
			singleObj = new Multiply();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public double execute(double in1, double in2) {
		return in1 * in2;
	}

	@Override
	public double execute(double in1, int in2) {
		return in1 * in2;
	}

	@Override
	public double execute(int in1, double in2) {
		return in1 * in2;
	}

	@Override
	public double execute(int in1, int in2) {
		return in1 * in2;
	}

}
