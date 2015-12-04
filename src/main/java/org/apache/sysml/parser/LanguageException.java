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

package org.apache.sysml.parser;

import org.apache.sysml.api.DMLException;

/**
 * <p>Exception occurring at the Language level.</p>
 */
public class LanguageException extends DMLException 
{
	
	private static final long serialVersionUID = 1L;
	
    /**
     * @see java.lang.Exception#Exception()
     */
    public LanguageException() {
        super();
    }
    
    /**
     * @see java.lang.Exception#Exception(String)
     */
    public LanguageException(String message) {
        super(message);
    }
    
    /**
     * @see java.lang.Exception#Exception(Throwable)
     */
    public LanguageException(Throwable cause) {
        super(cause);
    }
    
    /**
     * @see java.lang.Exception#Exception(String, Throwable)
     */
    public LanguageException(String message, Throwable cause) {
        super(message, cause);
    }
    
    public LanguageException(String message, String code) {
        super(code + ERROR_MSG_DELIMITER + message);
    }
    
    public static class LanguageErrorCodes {
    	public static final String UNSUPPORTED_EXPRESSION = "Unsupported Expression";
    	public static final String INVALID_PARAMETERS = "Invalid Parameters";
    	public static final String UNSUPPORTED_PARAMETERS = "Unsupported Parameters";
    	public static final String GENERIC_ERROR = "Language Syntax Error";
    }

}
