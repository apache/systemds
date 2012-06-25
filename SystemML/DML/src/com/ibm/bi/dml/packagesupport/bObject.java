package com.ibm.bi.dml.packagesupport;

/**
 * Class to represent an object.
 * 
 */

public class bObject extends FIO {


	private static final long serialVersionUID = 314464073593116450L;
	Object o;

	/**
	 * constructor that takes object as param
	 * 
	 * @param o
	 */
	public bObject(Object o) {
		super(Type.Object);
		this.o = o;
	}

	/**
	 * Method to retrieve object.
	 * 
	 * @return
	 */
	public Object getObject() {
		return o;
	}

}
