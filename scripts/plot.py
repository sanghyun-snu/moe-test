# visualize_sparse_compression.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 스타일 설정 (선택사항)
sns.set(style="whitegrid")

# CSV 파일 불러오기 (같은 디렉토리에 저장되어 있어야 함)
df = pd.read_csv("results.csv")  # 실제 파일 이름으로 변경하세요

# 예제 1: sparsity vs. 각 포맷의 시간
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='sparsity', y='dense_time_ms', label='Dense')
sns.lineplot(data=df, x='sparsity', y='csr_time_ms', label='CSR')
sns.lineplot(data=df, x='sparsity', y='lz4_time_ms', label='LZ4')
plt.title("Sparsity vs. Time (ms)")
plt.xlabel("Sparsity")
plt.ylabel("Time (ms)")
plt.legend()
plt.tight_layout()
plt.savefig("sparsity_vs_time.png")  # 저장
plt.clf()

# 예제 2: CSR Ratio vs. Sparsity (스레드 별)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='sparsity', y='csr_ratio', hue='threads', marker='o')
plt.title("CSR Ratio vs. Sparsity (by Threads)")
plt.xlabel("Sparsity")
plt.ylabel("CSR Ratio")
plt.tight_layout()
plt.savefig("csr_ratio_vs_sparsity.png")
plt.clf()

# 예제 3: 스레드별 압축 효율 비교 (박스 플롯)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='threads', y='total_efficiency')
plt.title("Total Compression Efficiency by Threads")
plt.xlabel("Threads")
plt.ylabel("Total Efficiency (%)")
plt.tight_layout()
plt.savefig("total_efficiency_by_threads.png")
plt.clf()

print("그래프가 성공적으로 저장되었습니다.")
