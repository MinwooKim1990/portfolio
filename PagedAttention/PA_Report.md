# Report on Comparing Standard Attention and Paged Attention

This report presents a concise overview and analysis of two attention mechanisms—**Standard Attention** and **Paged Attention**—implemented in Python using NumPy. The purpose of this comparison is to observe the differences in memory usage and execution time between the two methods and to visualise key results.

---

## 1. Introduction

In modern deep learning models such as Transformers, the self-attention mechanism plays a pivotal role in capturing relationships between tokens in a sequence. However, when dealing with long input sequences, the computational overhead and memory usage can become excessive. To mitigate this, **Paged Attention** is introduced as a strategy to process the sequence in smaller blocks (pages), thereby reducing peak memory usage.

---

## 2. Implementation Details

Two variants of the attention mechanism have been implemented in the provided code:

1. **Standard Attention**  
   - Computes the dot-product attention across the entire input sequence at once.  
   - Consists of three components: **Queries (Q)**, **Keys (K)**, and **Values (V)**.
   - The attention weights are obtained by:  
     \[
     \text{scores} = \frac{QK^\top}{\sqrt{d_k}}, \quad
     \text{attention\_weights} = \text{softmax}(\text{scores})
     \]
   - The output is then:  
     \[
     \text{output} = \text{attention\_weights} \times V
     \]

2. **Paged Attention**  
   - Splits the sequence into manageable blocks (pages) and calculates attention within each block.  
   - For each block of queries, `Q_block`, it attends over the corresponding portion of keys and values.
   - Memory usage per block is reduced since only parts of the sequence are processed at any one time.

### Softmax Function

The softmax function is defined as:

\[
\text{softmax}(x) = \frac{e^{x - \max(x)}}{\sum e^{x - \max(x)}}
\]

This normalises the attention scores into a probability distribution, ensuring all weights sum to 1.

---

## 3. Code Summary

Below is a high-level outline of the code structure:

- **`standard_attention(Q, K, V)`**:  
  Returns the attended output using the entire sequence at once.

- **`paged_attention(Q, K, V, block_size)`**:  
  Processes the input in blocks to reduce peak memory usage.

- **`softmax(x)`**:  
  Applies the softmax transformation for stable numerical computation.

- **`measure_performance(func, *args, **kwargs)`**:  
  Measures execution time and peak memory usage for a given function, using `time` and `tracemalloc`.

- **Main script**:
  1. Generates random `Q`, `K`, and `V`.
  2. Compares standard and paged attention mechanisms by measuring:
     - Execution time
     - Peak memory usage  
  3. Visualises:
     - Attention pattern (first 20 tokens)
     - Memory usage comparison
     - Execution time comparison
     - Memory usage per block in paged attention

---

## 4. Visualisation and Results
![Paged Attention comparison](https://github.com/user-attachments/assets/d66c7ca3-b397-400f-9a0e-60416761cdc1)
Four plots have been generated to illustrate the behaviour of the two attention methods:

1. **Attention Pattern (First 20 tokens)**  
   Displays the attention weights for the first 20 tokens (queries vs. keys). Darker cells indicate lower attention weights, whereas brighter (yellow) cells indicate higher attention weights.

2. **Peak Memory Usage Comparison**  
   A bar chart comparing the maximum memory consumption of:
   - **Standard Attention** (∼0.58 MB)
   - **Paged Attention** (∼0.12 MB)

3. **Execution Time Comparison**  
   A bar chart showing the time spent by:
   - **Standard Attention** (∼0.0000 s)
   - **Paged Attention** (∼0.0006 s)

4. **Paged Attention: Memory Usage per Block**  
   A line plot indicating how memory usage remains relatively small and consistent across each block during paged attention.

### Numerical Summary

| Metric | Standard Attention | Paged Attention |
| --- | --- | --- |
| Execution Time | 0.0000 s | 0.0006 s |
| Memory Usage | 0.58 MB | 0.12 MB |
- Memory Reduction: 78.7%


From the above, we see that:

- **Standard Attention** was notably faster (almost instantaneous) but consumed more memory.
- **Paged Attention** took slightly longer, yet used significantly less memory—resulting in a memory reduction of about **78.7%** relative to standard attention.

---

## 5. Discussion

1. **Memory Efficiency**:  
   Paged attention shows a clear advantage in terms of memory usage, making it suitable for long sequences where memory resources are constrained.

2. **Execution Time**:  
   Standard attention can be faster when the sequence length is within manageable limits, due to not having to iterate over smaller blocks. However, as sequence length grows, the difference in memory usage can be a deciding factor for real-world applications.

3. **Trade-offs**:  
   - **Paged Attention** provides scalability at the cost of a slight increase in computation time.  
   - **Standard Attention** is simpler and can be faster for moderate sequence lengths, but risks out-of-memory issues for longer sequences.

---

## 6. Conclusion

This comparison demonstrates the value of **Paged Attention** in scenarios where memory constraints are paramount. While **Standard Attention** may be faster for smaller sequences, **Paged Attention** scales better with sequence length and provides a substantial reduction in peak memory usage. 

For large-scale language models and resource-limited environments, this trade-off is crucial for enabling attention-based models to process longer contexts without running out of memory.

---

## 7. Future Work

1. **Block Overlaps**: Investigating the possibility of overlapping blocks to capture cross-block dependencies more effectively.  
2. **Dynamic Block Sizing**: Implementing an adaptive mechanism for adjusting `block_size` based on context length and available system resources.  
3. **Optimised GPU Implementations**: Porting these methods to GPU frameworks (e.g., PyTorch, TensorFlow) for faster parallel computation.  

---

**References**
- [Kwon et al. (2023)](https://arxiv.org/abs/2309.06180) *Efficient Memory Management for Large Language Model Serving with PagedAttention*