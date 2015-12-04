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

package org.apache.sysml.hops.rewrite;

import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.apache.sysml.hops.HopsException;
import org.apache.sysml.parser.StatementBlock;

/**
 * Base class for all hop rewrites in order to enable generic
 * application of all rules.
 * 
 */
public abstract class StatementBlockRewriteRule 
{

	protected static final Log LOG = LogFactory.getLog(StatementBlockRewriteRule.class.getName());
		
	/**
	 * Handle an arbitrary statement block. Specific type constraints have to be ensured
	 * within the individual rewrites.
	 * 
	 * @param sb
	 * @param sate
	 * @return
	 * @throws HopsException
	 */
	public abstract ArrayList<StatementBlock> rewriteStatementBlock( StatementBlock sb, ProgramRewriteStatus sate ) 
		throws HopsException;
	
}
