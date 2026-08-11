#!/usr/bin/env bash
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
#
# Regenerates the certificates in src/test/resources/cert, used by the federated SSL tests. The generated
# files are checked in, so this script only has to be run if a certificate has to be replaced, for instance
# because it expired or because the used algorithms are no longer accepted.
#
# The certificates are only used by tests, they authenticate nothing outside of a test run.
#
# Usage: src/test/scripts/functions/federated/io/generate-certificates.sh

set -euo pipefail
# from src/test/scripts/functions/federated/io up to src/test, then into the certificate directory
cd "$(dirname "$0")/../../../../resources/cert"

PW=changeit
# ~100 years, the certificates are checked in and should not have to be replaced because of expiry
DAYS=36500

rm -f ./*.pem ./*.p12 ./*.csr

# The authority the coordinator trusts, the basic constraint marks it as allowed to sign other certificates.
keytool -genkeypair -alias ca -dname "CN=SystemDS Test CA" -ext bc:c -keyalg RSA -keysize 2048 \
	-validity $DAYS -storetype PKCS12 -keystore ca.p12 -storepass $PW -keypass $PW
keytool -exportcert -alias ca -keystore ca.p12 -storepass $PW -rfc -file ca-cert.pem

# A second authority, unrelated to the one above and unknown to the coordinator.
keytool -genkeypair -alias ca -dname "CN=SystemDS Untrusted Test CA" -ext bc:c -keyalg RSA -keysize 2048 \
	-validity $DAYS -storetype PKCS12 -keystore untrusted-ca.p12 -storepass $PW -keypass $PW
keytool -exportcert -alias ca -keystore untrusted-ca.p12 -storepass $PW -rfc -file untrusted-ca-cert.pem

# Worker certificates signed by the first authority. The common name and the subject alternative names have
# to match the host the coordinator connects to, otherwise the host name verification rejects the worker.
gen_worker() {
	local name=$1 dname=$2 san=$3
	keytool -genkeypair -alias worker -dname "$dname" -keyalg RSA -keysize 2048 -validity $DAYS \
		-storetype PKCS12 -keystore "$name.p12" -storepass $PW -keypass $PW
	keytool -certreq -alias worker -keystore "$name.p12" -storepass $PW -file "$name.csr"
	keytool -gencert -alias ca -keystore ca.p12 -storepass $PW -infile "$name.csr" \
		-outfile "$name-cert.pem" -rfc -validity $DAYS -ext "san=$san"
	# the chain presented by the worker, leaf certificate first
	cat ca-cert.pem >> "$name-cert.pem"
	# the private key, unencrypted PKCS#8 as read by the worker
	openssl pkcs12 -in "$name.p12" -nocerts -nodes -passin "pass:$PW" \
		| sed -n '/BEGIN PRIVATE KEY/,/END PRIVATE KEY/p' > "$name-key.pem"
}

gen_worker localhost "CN=localhost" "dns:localhost,ip:127.0.0.1"
gen_worker otherhost "CN=other.example.com" "dns:other.example.com"

rm -f ./*.p12 ./*.csr
echo "Generated:"
ls -1 ./*.pem
