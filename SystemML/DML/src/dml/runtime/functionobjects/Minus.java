package dml.runtime.functionobjects;

public class Minus extends ValueFunction {

	private static Minus singleObj = null;
	
	private Minus() {
		// nothing to do here
	}
	
	public static Minus getMinusFnObject() {
		if ( singleObj == null )
			singleObj = new Minus();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public double execute(double in1, double in2) {
		return in1 - in2;
	}

	@Override
	public double execute(double in1, int in2) {
		return in1 - in2;
	}

	@Override
	public double execute(int in1, double in2) {
		return in1 - in2;
	}

	@Override
	public double execute(int in1, int in2) {
		return in1 - in2;
	}

}
