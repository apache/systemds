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

package org.apache.sysds.runtime.controlprogram.federated;

import java.io.Serializable;
import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.atomic.LongAdder;

import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.privacy.CheckedConstraintsLog;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;

public class FederatedResponse implements Serializable {
	private static final long serialVersionUID = 3142180026498695091L;
	
	public enum ResponseType {
		SUCCESS,
		SUCCESS_EMPTY,
		ERROR,
	}
	
	private ResponseType _status;
	private Object[] _data;
	private Map<PrivacyLevel,LongAdder> checkedConstraints;
	
	public FederatedResponse(ResponseType status) {
		this(status, null);
	}
	
	public FederatedResponse(ResponseType status, Object[] data) {
		_status = status;
		_data = data;
		if( _status == ResponseType.SUCCESS && data == null )
			_status = ResponseType.SUCCESS_EMPTY;
	}
	
	public FederatedResponse(FederatedResponse.ResponseType status, Object data) {
		_status = status;
		_data = new Object[] {data};
		if(_status == ResponseType.SUCCESS && data == null)
			_status = ResponseType.SUCCESS_EMPTY;
	}
	
	public boolean isSuccessful() {
		return _status != ResponseType.ERROR;
	}
	
	public String getErrorMessage() {
		if (_data[0] instanceof Throwable )
			return ExceptionUtils.getFullStackTrace( (Throwable) _data[0] );
		else if (_data[0] instanceof String)
			return (String) _data[0];
		else return "No readable error message";
	}
	
	public Object[] getData() throws Exception {
		updateCheckedConstraintsLog();
		if ( !isSuccessful() )
			throwExceptionFromResponse(); 
		return _data;
	}

	/**
	 * Checks the data object array for exceptions that occurred in the federated worker
	 * during handling of request. 
	 * @throws Exception the exception retrieved from the data object array 
	 *  or DMLRuntimeException if no exception is provided by the federated worker.
	 */
	public void throwExceptionFromResponse() throws Exception {
		for ( Object potentialException : _data){
			if (potentialException != null && (potentialException instanceof Exception) ){
				throw (Exception) potentialException;
			}
		}
		String errorMessage = getErrorMessage();
		if (getErrorMessage() != "No readable error message")
			throw new DMLRuntimeException(errorMessage);
		else
			throw new DMLRuntimeException("Unknown runtime exception in handling of federated request by federated worker.");
	}

	/**
	 * Set checked privacy constraints in response if the provided map is not empty.
	 * If the map is empty, it means that no privacy constraints were found.
	 * @param checkedConstraints map of checked constraints from the PrivacyMonitor
	 */
	public void setCheckedConstraints(Map<PrivacyLevel,LongAdder> checkedConstraints){
		if ( checkedConstraints != null && !checkedConstraints.isEmpty() ){
			this.checkedConstraints = new EnumMap<>(PrivacyLevel.class);
			this.checkedConstraints.putAll(checkedConstraints);
		}
	}

	public void updateCheckedConstraintsLog(){
		if ( checkedConstraints != null && !checkedConstraints.isEmpty() )
			CheckedConstraintsLog.addCheckedConstraints(checkedConstraints);
	}
}
