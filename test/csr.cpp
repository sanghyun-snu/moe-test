#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <lz4.h>

using namespace std;
using namespace chrono;

struct CSRMatrix {
    int rows, cols;
    vector<float> values;
    vector<int> col_indices;
    vector<int> row_ptr;
    vector<char> compressed_col_indices;  // 압축된 열 인덱스를 저장
    int compressed_size = 0;  // 압축 크기
};

vector<vector<float>> generate_dense_matrix(int rows, int cols, float sparsity) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    uniform_real_distribution<> val_dis(-1.0, 1.0);

    vector<vector<float>> mat(rows, vector<float>(cols, 0));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            if (dis(gen) > sparsity)
                mat[i][j] = val_dis(gen);
    return mat;
}

vector<vector<float>> generate_structured_sparse_matrix(int rows, int cols, float sparsity) {
    random_device rd;
    mt19937 gen(rd());
    int prune_cols = static_cast<int>(cols * sparsity);

    vector<int> indices(cols);
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), gen);

    vector<bool> keep(cols, true);
    for (int i = 0; i < prune_cols; ++i)
        keep[indices[i]] = false;

    uniform_real_distribution<> val_dis(-1.0, 1.0);
    vector<vector<float>> mat(rows, vector<float>(cols, 0));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            if (keep[j])
                mat[i][j] = val_dis(gen);
    return mat;
}

CSRMatrix dense_to_csr(const vector<vector<float>>& mat) {
    CSRMatrix csr;
    csr.rows = mat.size();
    csr.cols = mat[0].size();
    csr.row_ptr.push_back(0);

    for (const auto& row : mat) {
        for (int j = 0; j < row.size(); ++j) {
            if (row[j] != 0.0f) {
                csr.values.push_back(row[j]);
                csr.col_indices.push_back(j);
            }
        }
        csr.row_ptr.push_back(csr.values.size());
    }

    // 열 인덱스를 미리 압축
    int original_size = csr.col_indices.size() * sizeof(int);
    int max_compressed_size = LZ4_compressBound(original_size);
    csr.compressed_col_indices.resize(max_compressed_size);
    csr.compressed_size = LZ4_compress_default(
        (char*)csr.col_indices.data(),
        csr.compressed_col_indices.data(),
        original_size,
        max_compressed_size
    );

    return csr;
}

vector<float> csr_matvec(const CSRMatrix& csr, const vector<float>& x) {
    vector<float> y(csr.rows, 0.0f);
    for (int i = 0; i < csr.rows; ++i)
        for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; ++j)
            y[i] += csr.values[j] * x[csr.col_indices[j]];
    return y;
}

vector<float> csr_matvec_opt(const CSRMatrix& csr, const vector<float>& x) {
    vector<float> y(csr.rows, 0.0f);
    for (int i = 0; i < csr.rows; ++i)
        for (int j = csr.row_ptr[0]; j < csr.row_ptr[0 + 1]; ++j)
            y[i] += csr.values[j] * x[csr.col_indices[j]];
    return y;
}

vector<float> csr_matvec_with_compressed_col(const CSRMatrix& csr, const vector<float>& x) {
    // 복호화
    vector<int> decompressed(csr.col_indices.size());
    LZ4_decompress_safe(
        csr.compressed_col_indices.data(),
        (char*)decompressed.data(),
        csr.compressed_size,
        csr.col_indices.size() * sizeof(int)
    );

    // 연산
    vector<float> y(csr.rows, 0.0f);
    for (int i = 0; i < csr.rows; ++i)
        for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; ++j)
            y[i] += csr.values[j] * x[decompressed[j]];
    return y;
}

int main() {
    int rows = 4000, cols = 4000;
    float sparsity = 0.5;
    vector<float> x(cols, 1.0f);

    // 1. Random Sparse
    auto dense1 = generate_dense_matrix(rows, cols, sparsity);
    auto csr1 = dense_to_csr(dense1);

    cout << "== Random Sparse Matrix ==\n";
    auto start = high_resolution_clock::now();
    auto y1 = csr_matvec(csr1, x);
    auto end = high_resolution_clock::now();
    cout << "CSR matvec: " << duration_cast<microseconds>(end - start).count() << " us\n";

    start = high_resolution_clock::now();
    auto y2 = csr_matvec_opt(csr1, x);
    end = high_resolution_clock::now();
    cout << "CSR matvec opt: " << duration_cast<microseconds>(end - start).count() << " us\n";

    start = high_resolution_clock::now();
    auto y3 = csr_matvec_with_compressed_col(csr1, x);
    end = high_resolution_clock::now();
    cout << "CSR with compressed col indices (includes decompression): " << duration_cast<microseconds>(end - start).count() << " us\n";

    // 2. Structured Sparse
    auto dense2 = generate_structured_sparse_matrix(rows, cols, sparsity);
    auto csr2 = dense_to_csr(dense2);

    cout << "\n== Structured Sparse Matrix ==\n";
    start = high_resolution_clock::now();
    auto y4 = csr_matvec(csr2, x);
    end = high_resolution_clock::now();
    cout << "CSR matvec: " << duration_cast<microseconds>(end - start).count() << " us\n";

    start = high_resolution_clock::now();
    auto y5 = csr_matvec_with_compressed_col(csr2, x);
    end = high_resolution_clock::now();
    cout << "CSR with compressed col indices (includes decompression): " << duration_cast<microseconds>(end - start).count() << " us\n";

    return 0;
}
