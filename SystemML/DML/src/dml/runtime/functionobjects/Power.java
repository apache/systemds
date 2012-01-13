package dml.runtime.functionobjects;

// Singleton class

public class Power extends ValueFunction {

	private static Power singleObj = null;
	
	private Power() {
		// nothing to do here
	}
	
	public static Power getPowerFnObject() {
		if ( singleObj == null )
			singleObj = new Power();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public double execute(double in1, double in2) {
		return Math.pow(in1, in2); 
	}

	@Override
	public double execute(double in1, int in2) {
		return Math.pow(in1, (double)in2); 
	}

	@Override
	public double execute(int in1, double in2) {
		return Math.pow((double)in1, in2); 
	}

	@Override
	public double execute(int in1, int in2) {
		return Math.pow((double)in1, (double)in2);
	}

}
