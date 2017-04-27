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
#include "preload_systemml.h" 
#include <cstdlib>
 
//  g++ -o libpreload_systemml-linux-x86_64.so preload_systemml.cpp  -I$JAVA_HOME/include -I$JAVA_HOME/include/linux -lm -ldl -O3 -shared -fPIC
JNIEXPORT void JNICALL Java_org_apache_sysml_utils_EnvironmentHelper_setEnv(JNIEnv * env, jclass c, jstring jname, jstring jvalue) {
	const char* name = (env)->GetStringUTFChars(jname, NULL);
    	const char* value = (env)->GetStringUTFChars(jvalue,NULL);
#if defined _WIN32 || defined _WIN64 
	_putenv_s(name, value);
#else 
	setenv(name, value, 1);
#endif
	(env)->ReleaseStringUTFChars(jname, name); 
    	(env)->ReleaseStringUTFChars(jvalue, value);
}
 
 
