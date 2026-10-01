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

#include "cujava_logger.hpp"
#include <cstdarg>
#include <cstdio>

LogLevel Logger::currentLogLevel = LOG_ERROR;

void Logger::log(LogLevel level, const char *message, ...)
{
    if (level <= Logger::currentLogLevel)
    {
        va_list argp;
        va_start(argp, message);
        vfprintf(stdout, message, argp);
        va_end(argp);
        fflush(stdout);
    }
}

void Logger::setLogLevel(LogLevel level)
{
    Logger::currentLogLevel = level;
}
