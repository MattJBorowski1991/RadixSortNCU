# Top-P Sampling CUDA Implementation

## Build and Run

Compile the code:
```
nvcc test.cu top_p_sampling.cu -o test
```

Run all tests:
```
./test
```

Run the original failing test in isolation:
```
./test original
```

Run a specific test with custom vocab_size and p:
```
./test <vocab_size> <p>
```
Example:
```
./test 100 0.9
```
This runs the test with vocab_size=100 and p=0.9, using logits [0, 1, 2, ..., 99] and seed=123.

Note: The original failing test (vocab_size=3, logits=[1,2,3], p≈0.95) is run with `./test original` or as part of all tests.
