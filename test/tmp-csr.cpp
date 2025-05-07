#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <lz4.h>
#include <cstring>
#include <sched.h>
#include <omp.h>
#include <fstream>
#include <json.hpp>

using namespace std;
using namespace chrono;
using json = nlohmann::json;

void pin_to_cpu(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        perror("sched_setaffinity");
    } else {
        cout << "Pinned to CPU " << cpu_id << endl;
    }
}

void generate_dense_matrix(vector<float>& matrix, int rows, int cols, float sparsity) {
    default_random_engine rng(42);
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float r = dist(rng);
            matrix[i * cols + j] = (r < sparsity) ? 0.0f : dist(rng);
        }
    }
}

struct CSRMatrix {
    vector<float> values;
    vector<int> col_indices;
    vector<int> row_ptr;
};

CSRMatrix dense_to_csr(const vector<float>& dense, int rows, int cols) {
    CSRMatrix csr;
    csr.row_ptr.resize(rows + 1, 0);

    vector<int> nnz_per_row(rows, 0);
    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        int count = 0;
        for (int j = 0; j < cols; ++j) {
            if (dense[i * cols + j] != 0.0f)
                count++;
        }
        nnz_per_row[i] = count;
    }

    for (int i = 0; i < rows; ++i) {
        csr.row_ptr[i + 1] = csr.row_ptr[i] + nnz_per_row[i];
    }

    int nnz_total = csr.row_ptr[rows];
    csr.values.resize(nnz_total);
    csr.col_indices.resize(nnz_total);

    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        int offset = csr.row_ptr[i];
        for (int j = 0; j < cols; ++j) {
            float val = dense[i * cols + j];
            if (val != 0.0f) {
                csr.values[offset] = val;
                csr.col_indices[offset] = j;
                offset++;
            }
        }
    }

    return csr;
}

int compress_lz4(const char* input, int input_size, vector<char>& compressed) {
    int max_dst_size = LZ4_compressBound(input_size);
    compressed.resize(max_dst_size);
    int compressed_size = LZ4_compress_default(input, compressed.data(), input_size, max_dst_size);
    compressed.resize(compressed_size);
    return compressed_size;
}

void run_experiment(float sparsity, int rows, int cols, int num_threads, ofstream& csv) {
    omp_set_num_threads(num_threads);
    cout << "\n== Threads: " << num_threads << ", Sparsity: " << sparsity << " ==" << endl;

    vector<float> dense(rows * cols);
    generate_dense_matrix(dense, rows, cols, sparsity);

    auto start = high_resolution_clock::now();
    int dense_size = dense.size() * sizeof(float);
    vector<char> dense_compressed;
    int dense_compressed_size = compress_lz4(reinterpret_cast<const char*>(dense.data()), dense_size, dense_compressed);
    auto end = high_resolution_clock::now();
    auto dense_time_ms = duration_cast<nanoseconds>(end - start).count() / 1e6;

    start = high_resolution_clock::now();
    CSRMatrix csr = dense_to_csr(dense, rows, cols);
    end = high_resolution_clock::now();
    auto csr_convert_time = duration_cast<nanoseconds>(end - start).count() / 1e6;

    int csr_val_size = csr.values.size() * sizeof(float);
    int csr_idx_size = csr.col_indices.size() * sizeof(int);
    int csr_ptr_size = csr.row_ptr.size() * sizeof(int);
    int csr_total_size = csr_val_size + csr_idx_size + csr_ptr_size;

    vector<char> csr_blob(csr_total_size);
    memcpy(csr_blob.data(), csr.values.data(), csr_val_size);
    memcpy(csr_blob.data() + csr_val_size, csr.col_indices.data(), csr_idx_size);
    memcpy(csr_blob.data() + csr_val_size + csr_idx_size, csr.row_ptr.data(), csr_ptr_size);

    start = high_resolution_clock::now();
    vector<char> csr_compressed;
    int csr_compressed_size = compress_lz4(csr_blob.data(), csr_total_size, csr_compressed);
    end = high_resolution_clock::now();
    auto lz4_time_ms = duration_cast<nanoseconds>(end - start).count() / 1e6;

    float csr_ratio = 100.0 * csr_total_size / dense_size;
    float lz4_efficiency = 100.0 * (1.0 - (float)csr_compressed_size / csr_total_size);
    float total_efficiency = 100.0 * (1.0 - (float)csr_compressed_size / dense_size);

    csv << num_threads << "," << sparsity << "," << dense_size << "," << dense_compressed_size << "," << dense_time_ms << ","
        << csr_total_size << "," << csr_compressed_size << "," << csr_ratio << ","
        << lz4_efficiency << "," << total_efficiency << ","
        << csr_convert_time << "," << lz4_time_ms << endl;
}

// ... (상단 #include 및 정의는 동일)

int main() {
    pin_to_cpu(0);

    ifstream config_file("config.json");
    json config;
    config_file >> config;

    vector<int> thread_list = config["threads"].get<vector<int>>();
    vector<float> sparsity_list = config["sparsity_list"].get<vector<float>>();
    vector<vector<int>> sizes = config["matrix_sizes"]; // [[rows, cols], [rows, cols], ...]

    ofstream csv("results.csv");
    csv << "threads,rows,cols,sparsity,dense_size,dense_compressed_size,dense_time_ms,csr_size,csr_compressed_size,csr_ratio,lz4_efficiency,total_efficiency,csr_time_ms,lz4_time_ms\n";

    for (int threads : thread_list) {
        omp_set_num_threads(threads);
        cout << "\n==== Running with " << threads << " threads ====" << endl;

        for (auto& size_pair : sizes) {
            int rows = size_pair[0];
            int cols = size_pair[1];

            for (float sparsity : sparsity_list) {
                cout << "\n--- rows=" << rows << ", cols=" << cols << ", sparsity=" << sparsity << " ---" << endl;

                vector<float> dense(rows * cols);
                generate_dense_matrix(dense, rows, cols, sparsity);

                // Dense + LZ4
                auto start = high_resolution_clock::now();
                int dense_size = dense.size() * sizeof(float);
                vector<char> dense_compressed;
                int dense_compressed_size = compress_lz4(reinterpret_cast<const char*>(dense.data()), dense_size, dense_compressed);
                auto end = high_resolution_clock::now();
                auto dense_time_ms = duration_cast<nanoseconds>(end - start).count() / 1e6;

                // Dense -> CSR
                start = high_resolution_clock::now();
                CSRMatrix csr = dense_to_csr(dense, rows, cols);
                end = high_resolution_clock::now();
                auto csr_convert_time = duration_cast<nanoseconds>(end - start).count() / 1e6;

                int csr_val_size = csr.values.size() * sizeof(float);
                int csr_idx_size = csr.col_indices.size() * sizeof(int);
                int csr_ptr_size = csr.row_ptr.size() * sizeof(int);
                int csr_total_size = csr_val_size + csr_idx_size + csr_ptr_size;

                vector<char> csr_blob(csr_total_size);
                memcpy(csr_blob.data(), csr.values.data(), csr_val_size);
                memcpy(csr_blob.data() + csr_val_size, csr.col_indices.data(), csr_idx_size);
                memcpy(csr_blob.data() + csr_val_size + csr_idx_size, csr.row_ptr.data(), csr_ptr_size);

                start = high_resolution_clock::now();
                vector<char> csr_compressed;
                int csr_compressed_size = compress_lz4(csr_blob.data(), csr_total_size, csr_compressed);
                end = high_resolution_clock::now();
                auto lz4_time_ms = duration_cast<nanoseconds>(end - start).count() / 1e6;

                float csr_ratio = 100.0 * csr_total_size / dense_size;
                float lz4_efficiency = 100.0 * (1.0 - (float)csr_compressed_size / csr_total_size);
                float total_efficiency = 100.0 * (1.0 - (float)csr_compressed_size / dense_size);

                csv << threads << "," << rows << "," << cols << "," << sparsity << ","
                    << dense_size << "," << dense_compressed_size << "," << dense_time_ms << ","
                    << csr_total_size << "," << csr_compressed_size << "," << csr_ratio << ","
                    << lz4_efficiency << "," << total_efficiency << ","
                    << csr_convert_time << "," << lz4_time_ms << endl;
            }
        }
    }

    csv.close();
    return 0;
}
