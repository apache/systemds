/*
 * Copyright (C) IBM Corp. 2013.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
package com.ibm.biginsights.logging.jmx;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import org.apache.commons.lang.StringUtils;

public class Log4jConfig implements Log4jConfigMXBean{  
	
	public Logger getLogger(String logger){
		return Logger.getLogger(logger);
	}
  
	//Need more error handling
    public String getLogLevel(String logger) {  
        String level = "unavailable";  
  
        if (StringUtils.isNotBlank(logger)) {  
            Logger log = Logger.getLogger(logger);  
  
            if (log != null) {  
                level = log.getLevel().toString();  
            }  
        }  
        return level;  
    }  
    
    public void setLogLevel(String logger, String level) {  
        if (StringUtils.isNotBlank(logger)  &&  StringUtils.isNotBlank(level)) {  
            Logger log = Logger.getLogger(logger);  
  
            if (log != null) {  
                log.setLevel(Level.toLevel(level.toUpperCase()));  
            }  
        }  
    }  
}  
