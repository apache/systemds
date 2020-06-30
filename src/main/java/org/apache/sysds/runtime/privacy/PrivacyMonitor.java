/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.privacy;

import java.util.EnumMap;
import java.util.concurrent.atomic.LongAdder;

import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;

public class PrivacyMonitor 
{ 
	private static EnumMap<PrivacyLevel,LongAdder> checkedConstraints;

	static {
		checkedConstraints = new EnumMap<PrivacyLevel,LongAdder>(PrivacyLevel.class);
		for ( PrivacyLevel level : PrivacyLevel.values() ){
			checkedConstraints.put(level, new LongAdder());
		}
	}

	private static boolean checkPrivacy = false;

	public static EnumMap<PrivacyLevel,LongAdder> getCheckedConstraints(){
		return checkedConstraints;
	}

	private static void incrementCheckedConstraints(PrivacyLevel privacyLevel){
		if ( checkPrivacy ){
			if ( privacyLevel == null )
				throw new NullPointerException("Cannot increment checked constraints log: Privacy level is null.");
			checkedConstraints.get(privacyLevel).increment();
		}
			
	}

	public static void clearCheckedConstraints(){
		checkedConstraints.replaceAll((k,v)->new LongAdder());
	}

	public static void setCheckPrivacy(boolean checkPrivacyParam){
		checkPrivacy = checkPrivacyParam;
	}

	/**
	 * Throws DMLPrivacyException if data object is CacheableData and privacy constraint is set to private or private aggregation.
	 * @param dataObject input data object
	 * @return data object or data object with privacy constraint removed in case the privacy level was none. 
	 */
	public static Data handlePrivacy(Data dataObject){
		if ( dataObject instanceof CacheableData<?> ){
			PrivacyConstraint privacyConstraint = ((CacheableData<?>)dataObject).getPrivacyConstraint();
			if (privacyConstraint != null){
				PrivacyLevel privacyLevel = privacyConstraint.getPrivacyLevel();
				incrementCheckedConstraints(privacyLevel);
				switch(privacyLevel){
					case None:
						((CacheableData<?>)dataObject).setPrivacyConstraints(null);
						break;
					case Private:
					case PrivateAggregation:
						throw new DMLPrivacyException("Cannot share variable, since the privacy constraint of the requested variable is set to " + privacyLevel.name());
					default:
						throw new DMLPrivacyException("Privacy level " + privacyLevel.name() + " of variable not recognized");
				}
			}
		}
		return dataObject;
	}

	/**
	 * Throws DMLPrivacyException if privacy constraint of matrix object has level privacy.
	 * @param matrixObject input matrix object
	 * @return matrix object or matrix object with privacy constraint removed in case the privacy level was none.
	 */
	public static MatrixObject handlePrivacy(MatrixObject matrixObject){
		PrivacyConstraint privacyConstraint = matrixObject.getPrivacyConstraint();
		if (privacyConstraint != null){
			PrivacyLevel privacyLevel = privacyConstraint.getPrivacyLevel();
			incrementCheckedConstraints(privacyLevel);
			switch(privacyLevel){
				case None:
					matrixObject.setPrivacyConstraints(null);
					break;
				case Private:
					throw new DMLPrivacyException("Cannot share variable, since the privacy constraint of the requested variable is set to " + privacyLevel.name());
				case PrivateAggregation:
					break; 
				default:
					throw new DMLPrivacyException("Privacy level " + privacyLevel.name() + " of variable not recognized");
			}
		}
		return matrixObject;
	}
	
	/**
	 * Throw DMLPrivacyException if privacy is activated for the input variable
	 * @param input variable for which the privacy constraint is checked
	 */
	public static void handlePrivacyScalarOutput(CPOperand input, ExecutionContext ec) {
		Data data = ec.getVariable(input);
		if ( data != null && (data instanceof CacheableData<?>)){
			PrivacyConstraint privacyConstraintIn = ((CacheableData<?>) data).getPrivacyConstraint();
			if ( privacyConstraintIn != null ) {
				incrementCheckedConstraints(privacyConstraintIn.getPrivacyLevel());
				if ( privacyConstraintIn.getPrivacyLevel() == PrivacyLevel.Private ){
					throw new DMLPrivacyException("Privacy constraint cannot be propagated to scalar for input " + input.getName());
				}
			}
		}
	}
}
