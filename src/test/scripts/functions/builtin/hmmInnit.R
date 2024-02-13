#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

args <- commandArgs(TRUE)
hiddenstates_count <- as.integer(args[1])
observation_count <- as.integer(args[0])

start_prob <- rep(1 / hiddenstates_count, hiddenstates_count)
transition_prob <- 0.3 * diag(hiddenstates_count) + array(0.7 / (hiddenstates_count), c(hiddenstates_count, hiddenstates_count)) # nolint: line_length_linter.
emission_prob <- array(1 / (observation_count), c(hiddenstates_count, observation_count)) # nolint: line_length_linter.

writeMM(as(start_prob, "start_prob"), paste(args[2], "start_prob", sep = ""))
writeMM(as(transition_prob, "transition_prob"), paste(args[2], "transition_prob", sep="")) # nolint: line_length_linter.
writeMM(as(emission_prob, "emission_prob"), paste(args[2], "emission_prob", sep = "")) # nolint: line_length_linter.
