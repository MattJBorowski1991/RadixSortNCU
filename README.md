# Top-P Sampling CUDA Implementation

## Build and Run

Compile the code:
```
nvcc -Iinclude -Iprof tests/test_batch_normal.cu top_p.cu src/softmax.cu src/sort.cu src/reverse.cu src/nucleus.cu src/sample.cu -o test_batch_normal```

Run a tests:
```
./test_batch_normal --batches 64 --vocab_size 4096 --variance 12.0 --seed 78```



