#include "he.h"
#include "libhe.h"

#ifdef _WIN32
#include <winsock.h>
#else
#include <arpa/inet.h>
#endif

unique_ptr<istream> get_stream(JNIEnv* env, jbyteArray ary) {
    size_t size = env->GetArrayLength(ary);
    jbyte* data = env->GetByteArrayElements(ary, NULL);

    // FIXME: this copies string data once. maybe implement a custom stream
    // idea: implement a custom stream that wraps a jbyteArray, which calls ReleaseByteArrayElements in its d'tor
    string data_s = string(reinterpret_cast<char*>(data), size);
    unique_ptr<istream> ret = std::make_unique<istringstream>(std::move(data_s));
    env->ReleaseByteArrayElements(ary, data, JNI_ABORT);
    return ret;
}

jbyteArray allocate_byte_array(JNIEnv* env, ostringstream& stream) {
    string data = stream.str(); // FIXME: this copies string content. maybe implement custom ostream
    jbyteArray ret = env->NewByteArray(data.size());
    env->SetByteArrayRegion(ret, 0, data.size(), reinterpret_cast<jbyte*>(data.data()));
    return ret;
}

void my_assert(bool assertion, const char* message = "Assertion failed") {
    if (!assertion) {
        throw logic_error(message);
    }
}

template<typename T> jbyteArray serialize(JNIEnv* env, T& object) {
    ostringstream ss;
    object.save(ss);
    return allocate_byte_array(env, ss);
}

void serialize_uint32_t(ostream& ss, uint32_t n) {
    n = htonl(n);
    ss.write(reinterpret_cast<char*>(&n), sizeof(n));
}

uint32_t deserialize_uint32_t(istream& ss) {
    uint32_t ret;
    ss.read(reinterpret_cast<char*>(&ret), sizeof(ret));
    ret = ntohl(ret);
    return ret;
}

Ciphertext deserialize_ciphertext(istream& ss, const SEALContext& context) {
    Ciphertext ret;
    ret.load(context, ss);
    return ret;
}

void serialize_plaintext(ostream& ss, Plaintext plaintext) {
    plaintext.save(ss);
}

template<typename T> T deserialize_unsafe(JNIEnv* env, const SEALContext& context, jbyteArray serialized_object) {
    auto ss = get_stream(env, serialized_object);
    T deserialized;
    deserialized.unsafe_load(context, *ss); // necessary bc partial public keys are not valid public keys
    return deserialized;
}

template<typename T> T deserialize(JNIEnv* env, const SEALContext& context, jbyteArray serialized_object) {
    auto ss = get_stream(env, serialized_object);
    T deserialized;
    deserialized.load(context, *ss); // necessary bc partial public keys are not valid public keys
    return deserialized;
}

JNIEXPORT jlong JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_initClient
  (JNIEnv* env, jclass, jbyteArray a_ary) {
    double scale = pow(2.0, 40);
    GlobalState gs(scale);

    // copy a to global state
    size_t byte_size = env->GetArrayLength(a_ary);
    my_assert(byte_size % sizeof(uint64_t) == 0);
    size_t size = byte_size / sizeof(uint64_t);
    uint64_t* a = reinterpret_cast<uint64_t*>(env->GetByteArrayElements(a_ary, NULL));
    gsl::span<uint64_t > new_a(a, size);

    vector<uint64_t> new_a_buf;
    new_a_buf.assign(new_a.begin(), new_a.end());
    gs.a.set_data(new_a_buf);

    // release a without back-copy
    env->ReleaseByteArrayElements(a_ary, reinterpret_cast<jbyte*>(a), JNI_ABORT);

    Client* client = new Client(gs);
    return reinterpret_cast<jlong>(client);
}


JNIEXPORT jbyteArray JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_generatePartialPublicKey
  (JNIEnv* env, jclass, jlong client_ptr) {
    Client* client = reinterpret_cast<Client*>(client_ptr);
    return serialize(env, client->partial_public_key().data());
}


JNIEXPORT void JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_setPublicKey
  (JNIEnv* env, jclass, jlong client_ptr, jbyteArray serialized_public_key) {
    Client* client = reinterpret_cast<Client*>(client_ptr);
    client->set_public_key(deserialize<PublicKey>(env, client->context(), serialized_public_key));
}


JNIEXPORT jbyteArray JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_encrypt
  (JNIEnv* env, jclass, jlong client_ptr, jdoubleArray jdata) {
    Client* client = reinterpret_cast<Client*>(client_ptr);
    size_t slot_count = get_slot_count(client->context());
    size_t num_data = env->GetArrayLength(jdata);
    const double* data = static_cast<const double*>(env->GetDoubleArrayElements(jdata, NULL));

    std::ostringstream ss;
    // write chunk size
    uint32_t num_chunks = (num_data - 1) / slot_count + 1;
    serialize_uint32_t(ss, num_chunks);
    for (size_t i = 0; i < num_chunks; i++) {
        size_t offset = slot_count * i;
        size_t length = min(slot_count, num_data-offset);
        gsl::span<const double> data_span(&data[offset], length);
        Ciphertext encrypted_chunk = client->encrypted_data(data_span);
        encrypted_chunk.save(ss);
    }
    env->ReleaseDoubleArrayElements(jdata, const_cast<jdouble*>(data), JNI_ABORT);
    return allocate_byte_array(env, ss);
}


JNIEXPORT jbyteArray JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_partiallyDecrypt
  (JNIEnv* env, jclass, jlong client_ptr, jbyteArray serialized_ciphertexts) {
    Client* client = reinterpret_cast<Client*>(client_ptr);
    auto input = get_stream(env, serialized_ciphertexts);
    std::ostringstream ss;

    // read num of chunks
    uint32_t num_chunks = deserialize_uint32_t(*input);

    // write chunk size
    serialize_uint32_t(ss, num_chunks);
    for (int i = 0; i < num_chunks; i++) {
        Ciphertext ciphertext = deserialize_ciphertext(*input, client->context());
        Plaintext plaintext = client->partial_decryption(ciphertext);
        plaintext.save(ss);
    }

    return allocate_byte_array(env, ss);
}


JNIEXPORT jlong JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_initServer
  (JNIEnv *, jclass) {
    double scale = pow(2.0, 40);
    GlobalState gs(scale);
    Server* server = new Server(gs);
    return reinterpret_cast<jlong>(server);
}


JNIEXPORT jbyteArray JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_generateA
  (JNIEnv* env, jclass, jlong server_ptr) {
    Server* server = reinterpret_cast<Server*>(server_ptr);
    uint64_t* data = server->a().data();
    size_t size = server->a().size() * sizeof(data[0]) / sizeof(jbyte);
    jbyteArray ret = env->NewByteArray(size);
    env->SetByteArrayRegion(ret, 0, size, reinterpret_cast<jbyte*>(data));
    return ret;
}


JNIEXPORT jbyteArray JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_aggregatePartialPublicKeys
  (JNIEnv* env, jclass, jlong server_ptr, jobjectArray partial_public_keys_serialized) {
    Server* server = reinterpret_cast<Server*>(server_ptr);
    size_t num_partial_public_keys = env->GetArrayLength(partial_public_keys_serialized);
    std::vector<Ciphertext> partial_public_keys;
    partial_public_keys.reserve(num_partial_public_keys);

    for (int i = 0; i < num_partial_public_keys; i++) {
        jbyteArray j_data = static_cast<jbyteArray>(env->GetObjectArrayElement(partial_public_keys_serialized, i));
        partial_public_keys.push_back(deserialize_unsafe<Ciphertext>(env, server->context(), j_data));
        env->DeleteLocalRef(j_data);
    }

    server->accumulate_partial_public_keys(gsl::span(partial_public_keys));
    return serialize(env, server->public_key());
}


JNIEXPORT jbyteArray JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_accumulateCiphertexts
  (JNIEnv* env, jclass, jlong server_ptr, jobjectArray ciphertexts_serialized) {
    Server* server = reinterpret_cast<Server*>(server_ptr);
    size_t num_ciphertext_arys = env->GetArrayLength(ciphertexts_serialized);

    // init streams
    vector<unique_ptr<istream>> buf;
    buf.reserve(num_ciphertext_arys);
    for (int i = 0; i < num_ciphertext_arys; i++) {
        jbyteArray j_data = static_cast<jbyteArray>(env->GetObjectArrayElement(ciphertexts_serialized, i));
        auto stream = get_stream(env, j_data);
        buf.emplace_back(std::move(stream));
        env->DeleteLocalRef(j_data);
    }

    // read lengths of ciphertext arys and check that they are all the same
    uint32_t num_slots = deserialize_uint32_t(*buf[0]);
    for (int i = 1; i < num_ciphertext_arys; i++) {
        my_assert(deserialize_uint32_t(*buf[i]) == num_slots);
    }

    // read ciphertexts in chunks and accumulate them
    ostringstream result;
    serialize_uint32_t(result, num_slots);
    for (int chunk_idx = 0; chunk_idx < num_slots; chunk_idx++) {
        vector<Ciphertext> ciphertexts;
        ciphertexts.reserve(num_ciphertext_arys);
        for (int i = 0; i < num_ciphertext_arys; i++) {
            Ciphertext deserialized;
            deserialized.load(server->context(), *buf[i]);
            ciphertexts.emplace_back(deserialized);
        }
        Ciphertext sum = server->sum_data(std::move(ciphertexts));
        sum.save(result);
    }

    return allocate_byte_array(env, result);
}


JNIEXPORT jdoubleArray JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_average
  (JNIEnv* env, jclass, jlong server_ptr, jbyteArray ciphertext_sum_serialized, jobjectArray partial_decryptions_serialized) {
    Server* server = reinterpret_cast<Server*>(server_ptr);
    size_t slot_size = get_slot_count(server->context());
    size_t num_plaintext_arys = env->GetArrayLength(partial_decryptions_serialized);

    // init streams
    vector<unique_ptr<istream>> buf;
    buf.reserve(num_plaintext_arys);
    for (int i = 0; i < num_plaintext_arys; i++) {
        jbyteArray j_data = static_cast<jbyteArray>(env->GetObjectArrayElement(partial_decryptions_serialized, i));
        auto stream = get_stream(env, j_data);
        buf.emplace_back(std::move(stream));
        env->DeleteLocalRef(j_data);
    }

    // read lengths of ciphertext arys and check that they are all the same
    uint32_t num_slots = deserialize_uint32_t(*buf[0]);
    for (int i = 1; i < num_plaintext_arys; i++) {
        my_assert(deserialize_uint32_t(*buf[i]) == num_slots, "number of plaintext slots is different");
    }

    auto encrypted_sum_stream = get_stream(env, ciphertext_sum_serialized);
    my_assert(deserialize_uint32_t(*encrypted_sum_stream) == num_slots, "number of ciphertext slots is different");

    // read ciphertexts in chunks and accumulate them
    jdoubleArray result = env->NewDoubleArray(num_slots * slot_size);
    for (int chunk_idx = 0; chunk_idx < num_slots; chunk_idx++) {
        Ciphertext encrypted_sum = deserialize_ciphertext(*encrypted_sum_stream, server->context());

        vector<Plaintext> partial_decryptions;
        partial_decryptions.reserve(num_plaintext_arys);
        for (int i = 0; i < num_plaintext_arys; i++) {
            Plaintext deserialized;
            deserialized.load(server->context(), *buf[i]);
            partial_decryptions.emplace_back(deserialized);
        }
        vector<double> averages = server->average(encrypted_sum, move(partial_decryptions));
        env->SetDoubleArrayRegion(result, chunk_idx*slot_size, averages.size(), averages.data());
    }

    return result;
}