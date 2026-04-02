# B2: How to integrate - Training Data

Це опис, як у подальшому підключити/взаємодіяти із нашим проєктом

## Як створити IBatchProvider з масиву токенiв

Напряму це зробити не можна, тому алгоритм дій такий:

```
// готовий масив токенів
int[] tokens = { 0, 1, 2, 3, 1, 2, 3, 1, 0, 8 };

// створення потрібних екземплярів класу
ITokenStream stream = new ArrayTokenStream(tokens);
BatchWindowSampler sampler = new BatchWindowSampler();

// створення batch provider, що має метод GetBatch для отримання батчів
IBatchProvider batchProvider = new TokenBatchProvider(stream, sampler);

// задання параметрів для створення конфігурації батчів
int batchSize = 8 (кількість пар батчів, що отримаємо)
int blockSize = 8192 (розмір контекстного вікна для TinyNN)

// створення конфігурації батчів
BatchConfig config = new BatchConfig(batchSize, blockSize);
Random rng = new Random();

Batch batch = GetBatch(config, rng);
```

таким чином, отримуємо Batch -> це об'єкт, що зберігає множину батчів
