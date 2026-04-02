# B2: How to integrate - Training Data

Це опис, як у подальшому підключити/взаємодіяти із нашим проєктом

## Як створити IBatchProvider з масиву токенiв

Напряму це зробити не можна, тому алгоритм дій такий:

```
// готовий масив токенів
int[] tokens = { 0, 1, 2, 3, 1, 2, 3, 1, 0, 8 };

// цей масив надає ITokenizator на виході, що приймає наша частина загального проєкту

// створення потрібних екземплярів класу
ITokenStream stream = new ArrayTokenStream(tokens);
BatchWindowSampler sampler = new BatchWindowSampler();

// створення batch provider, що має метод GetBatch для отримання батчів
IBatchProvider batchProvider = new TokenBatchProvider(stream, sampler);

// задання параметрів для створення конфігурації батчів
int batchSize = 8 
int blockSize = 8192 

// створення конфігурації батчів
BatchConfig batchConfig = new BatchConfig(batchSize, blockSize);
Random rng = new Random();

Batch batch = GetBatch(batchConfig, rng);
```

таким чином, отримуємо Batch -> це об'єкт, що зберігає множину батчів, що має:

int[][] Inputs -> це властивість, що відповідає за контекст
int[] Targets -> це властивість, що відповідає за ціль (що має видатись), використовується для навчання,
ціль є наступним токеном після контексту, що обирається "на рандом"

## Як створити TrainingConfig 

```
// задаємо параметри для конфігурації (значення є прикладом)
int epochs = 20
float learningRate = 0.01f
int checkpointInterval = 5

// створюємо об'єкт конфігурації
TrainingConfig trainingConfig = new TrainingConfig(epochs, learningRate, checkpointInterval)
```

## Як розпочати тренування моделі

```
ITrainingLoop trainingLoop = new TrainingLoop()

// виклик метода тренування
trainingLoop.Train(model, batchProvider, trainingConfig, batchConfig, tokens)
model -> це об'єкт типу ILanguageModel, у залежності від якого буде тренуватись конкретна модель
```

## Що є чим?

```
epochs -> кількість проходів по
batchSize -> кількість пар батчів
blockSize -> розмір контекстного вікна моделі
learningRate -> коефіцієнт сили оновлення ваг моделі
```

## Як це підключається до Trainer CLI 

1. CLI отримує токени
2. створює IBatchProvider
3. створює TrainingConfig
4. передає все у TrainingLoop

## Результат?

У результаті отримуємо тренування моделі



# B5: How to integrate - Neural

Це опис, як у подальшому підключити/взаємодіяти із нашим проєктом

## Як створити TinyNN із vocabSize

```
// розмір словника, що приходить від токенізатора
int vocabSize = tokenizer.VocabSize

// тип моделі
string modelKind = "tinynn"

// параметри моделі
int embeddingSize = 8192
int contextSize = 8

// створення конфігурації моделі + ваг
TinyNNConfig config = new TinyNNConfig(vocabSize, embeddingSize, contextSize)
TinyNNWeights weights = new TinyNNWeights(vocabSize, config)

// створення TinyNNModel
TinyNNModel model = new TinyNNModel(modelKind, vocabSize, config, weights)
```

## Що є чим?

EmbeddingSize -> це розмірність вектора, що представляє конкретний токен
ContextSize -> це розмір контекстного вікна моделі

## Формат checkpoint TinyNN

При серіалізації ToPayload() модель зберігається у такому форматі:
```
{
  "config": {
    "vocabSize": 1000,
    "embeddingSize": 32,
    "contextSize": 8
  },
  "embeddings": [[...], [...]],
  "outputWeights": [[...]],
  "outputBias": [...]
}
```

Щоб зберегти та відновити модель:
```
// Зберігання
var payload = model.ToPayload();
string json = JsonSerializer.Serialize(payload);
File.WriteAllText("model.json", json);

// Відновлення
string json = File.ReadAllText("model.json");
using var doc = JsonDocument.Parse(json);
var restoredModel = factory.CreateFromPayload(doc.RootElement);
```

Після завантаження модель відновлює ті самі параметри, що були збережені, тому при однаковому стані ваг вона видає ті самі результати

## Як створити Transformer із vocabSize

Найпростіший спосіб - через фабрику з одним параметром:

```
var factory = new TinyTransformerModelFactory();
var model = factory.Create(vocabSize: 1000);
```

Модель автоматично ініціалізує ваги випадковими значеннями.

Якщо потрібен deterministic result (наприклад, для тестів) то можна передати seed:

```
var model = factory.Create(vocabSize: 1000, seed: 42);
```

## Конфігурація

За замовчуванням використовуються такі значення:   
- EmbeddingSize: 16   
- HeadCount: 1   
- ContextSize: 8

Якщо дефолтні значення не підходять то можна вказати все явно:

```
var model = factory.Create(
    vocabSize: 1000,
    embeddingSize: 32,
    headCount: 2,
    contextSize: 16,
    seed: 42
);
```

Model automatically truncates context to only `ContextSize` last tokens if the input sequence is larger than the available context window.

## Формат checkpoint Transformer

При серіалізації `ToPayload()` модель зберігається у такому форматі:

```
{
  "config": {
    "vocabSize": 1000,
    "embeddingSize": 16,
    "headCount": 1,
    "contextSize": 8
  },
  "tokenEmbeddings": [[...], [...]],
  "wq": [[...]],
  "wk": [[...]],
  "wv": [[...]],
  "wo": [[...]],
  "ffn1": [[...]],
  "ffn1Bias": [...],
  "ffn2": [[...]],
  "ffn2Bias": [...],
  "outputW": [[...]],
  "outputBias": [...]
}
```

Щоб зберегти та відновити модель:

```
// Зберігання
var payload = model.ToPayload();
string json = JsonSerializer.Serialize(payload);
File.WriteAllText("model.json", json);

// Відновлення
string json = File.ReadAllText("model.json");
using var doc = JsonDocument.Parse(json);
var restoredModel = factory.CreateFromPayload(doc.RootElement);
```

Після завантаження модель видає ідентичні результати - це перевірено інтеграційними тестами.
