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

package org.apache.sysml.hops;

import org.apache.sysml.api.DMLException;

/**
 * <p>Exception occurring in the HOP level.</p>
 */
public class HopsException extends DMLException 
{
		
	private static final long serialVersionUID = 1L;
	
    public HopsException() {
        super();
    }

    public HopsException(String message) {
        super(message);
    }

    public HopsException(Throwable cause) {
        super(cause);
    }

    public HopsException(String message, Throwable cause) {
        super(message, cause);
    }

    /**
     * If the condition fails, print the message formatted with objects.
     * @param condition Condition to test
     * @param message Message to print if the condition fails
     * @param objects Objects to print with the message, as per String.format
     * @throws HopsException Thrown if condition is false
     */
    public static void check(boolean condition, String message, Object... objects) throws HopsException {
        if (!condition)
            throw new HopsException(String.format(message, objects));
    }
    /**
     * If the condition fails, print the Op and its Id, along with the message formatted with objects.
     * @param condition Condition to test
     * @param hop Hop to print as a cause of the problem, if the condition fails
     * @param message Message to print if the condition fails
     * @param objects Objects to print with the message, as per String.format
     * @throws HopsException Thrown if condition is false
     */
    public static void check(boolean condition, Hop hop, String message, Object... objects) throws HopsException {
        if (!condition)
            throw new HopsException(String.format(hop.getOpString()+" id="+hop.getHopID()+" "+message, objects));
    }
}
