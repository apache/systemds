package dml.packagesupport;

import java.io.Serializable;

/**
 * abstract class to represent all input and output objects for package
 * functions.
 * 
 * @author aghoting
 * 
 */

public abstract class FIO implements Serializable{


	private static final long serialVersionUID = 1189133371204708466L;
	Type type;

	/**
	 * Constructor to set type
	 * 
	 * @param type
	 */
	public FIO(Type type) {
		this.type = type;
	}

	/**
	 * Method to get type
	 * 
	 * @return
	 */
	public Type getType() {
		return type;
	}

}
