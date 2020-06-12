package org.apache.sysds.runtime.privacy;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.BiFunction;

import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;

public class CheckedConstraintsLog {
	private static ConcurrentHashMap<PrivacyLevel,LongAdder> checkedConstraintsTotal = new ConcurrentHashMap<PrivacyLevel,LongAdder>();
	private static BiFunction<LongAdder, LongAdder, LongAdder> mergeLongAdders = (v1, v2) -> {
		v1.add(v2.longValue() );
		return v1;
	};

	/**
	 * Adds checkedConstraints to the checked constraints total. 
	 * @param checkedConstraints constraints checked by federated worker
	 */
	public static void addCheckedConstraints(ConcurrentHashMap<PrivacyLevel,LongAdder> checkedConstraints){
		if ( checkedConstraints != null){
			checkedConstraints.forEach( 
			(key,value) -> checkedConstraintsTotal.merge( key, value, mergeLongAdders) );
		}
	}

	/**
	 * Remove all elements from checked constraints log.
	 */
	public static void reset(){
		checkedConstraintsTotal.clear();
	}

	public static ConcurrentHashMap<PrivacyLevel,LongAdder> getCheckedConstraints(){
		return checkedConstraintsTotal;
	}

	/**
	 * Get string representing all contents of the checked constraints log.
	 * @return string representation of checked constraints log.
	 */
	public static String display(){
		StringBuilder sb = new StringBuilder();
		checkedConstraintsTotal.forEach((k,v)->sb.append("\t" + k + ": " + v + "\n"));
		return sb.toString();
	}
}