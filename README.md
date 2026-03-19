# Lib.MathCore

## Опис

Бібліотека математичних операцій для Mini-ChatGPT.
Використовується для обробки логітів, обчислення функції втрат та вибору токенів.

## Класи

IMathOps інтерфейс для використання бібліотеки  
SoftmaxCalculator обчислює ймовірності з логітів  
LossCalculator обчислює функцію втрат  
ProbabilitySampler виконує вибір індексу  
MathOpsImpl реалізує інтерфейс IMathOps  

## Методи

Softmax(ReadOnlySpan<float> logits) перетворює логіти у ймовірності  
CrossEntropyLoss(ReadOnlySpan<float> logits, int target) обчислює помилку  
ArgMax(ReadOnlySpan<float> scores) повертає індекс максимуму  
SampleFromProbs(ReadOnlySpan<float> probs, Random rng) повертає випадковий індекс  

## Очікування інтеграції

Бібліотека використовується через інтерфейс IMathOps  
На вхід подаються логіти або ймовірності  
На вихід повертаються ймовірності або індекси токенів  