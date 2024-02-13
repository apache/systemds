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
library("Matrix")
library(HMM)

observation_sequence <- as.matrix(readMM(paste(args[1], "observation_sequence.mtx", sep = ""))) # nolint: line_length_linter.
start_prob <- as.matrix(readMM(paste(args[1], "start_prob.mtx", sep = "")))
transition_prob <- as.matrix(readMM(paste(args[1], "transition_prob.mtx", sep = ""))) # nolint: line_length_linter.
emission_prob <- as.matrix(readMM(paste(args[1], "emission_prob.mtx", sep = ""))) # nolint: line_length_linter.
num_iterations <- as.integer(args[2])
observation_count <- as.integer(args[3])
hiddenstates_count <- as.integer(args[4])

states <- 1:hiddenstates_count
symbols <- 1:observation_count
hmm <- list(States = states, Symbols = symbols, startProbs = start_prob, transProbs = transition_prob, emissionProbs = emission_prob) # nolint: line_length_linter.

c(hmm_updated, diff) <- baumWelch(hmm = hmm, observation = observation, maxIterations = num_iterations, delta = 0) # nolint: line_length_linter.

writeMM(as(hmm_updated$transProbs, "transition_prob"), paste(args[5], "transition_prob", sep = "")) # nolint: line_length_linter.
writeMM(as(hmm_updated$emissionProbs, "emission_prob"), paste(args[5], "emission_prob", sep = "")) # nolint: line_length_linter.
writeMM(as(diff, "diff_array"), paste(args[5], "diff_array", sep = "")) # nolint: line_length_linter.
