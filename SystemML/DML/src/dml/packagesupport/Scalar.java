package dml.packagesupport;

/**
 * Class to represent a scalar input/output.
 * 
 * @author aghoting
 * 
 */
public class Scalar extends FIO {


	private static final long serialVersionUID = 55239661026793046L;

	public enum ScalarType {
		Integer, Double, Boolean, Text
	};

	String value;
	ScalarType sType;

	/**
	 * Constructor to setup a scalar object.
	 * 
	 * @param t
	 * @param val
	 */
	public Scalar(ScalarType t, String val) {
		super(Type.Scalar);
		sType = t;
		value = val;
	}

	/**
	 * Method to get type of scalar.
	 * 
	 * @return
	 */
	public ScalarType getScalarType() {
		return sType;
	}

	/**
	 * Method to get value for scalar.
	 * 
	 * @return
	 */
	public String getValue() {
		return value;
	}

}
