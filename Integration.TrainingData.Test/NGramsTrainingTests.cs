using Contracts;
using Lib.Training;
using Lib.Training.Configuration;
using Lib.Training.Metrics;

namespace Integration.TrainingData.Test
{
    public class NGramsTrainingTests
    {
        [Test]
        public void BatchingAndTraining_CanRunOneEpoch_Bigram()
        {
            // Arrange
            int[] tokens = new int[] { 1, 2, 3, 2, 1, 2, 3};
            ILanguageModel model = new NGramModel(3);

            var trainingConfig = new TrainingConfig(1, 0, 1);
            TrainingLoop trainingLoop = new TrainingLoop();

            // Act
            TrainingMetrics metrics = trainingLoop.Train(model, null, trainingConfig, null, tokens);

            // Assert
            Assert.That(metrics, Is.Not.Null);
            Assert.That(metrics.Perplexity, Is.GreaterThan(0));
            Assert.That(float.IsFinite((float)metrics.Perplexity));
            Assert.That(metrics.NGramCount, Is.EqualTo(6));
            Assert.That(metrics.CurrentEpoch, Is.EqualTo(1));
        }

        [Test]
        public void BatchingAndTraining_CanRunOneEpoch_Trigram()
        {
            // Arrange
            int[] tokens = new int[] { 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1 };
            ILanguageModel model = new TrigramModel(3);

            var trainingConfig = new TrainingConfig(1, 0, 1);
            TrainingLoop trainingLoop = new TrainingLoop();

            // Act
            TrainingMetrics metrics = trainingLoop.Train(model, null, trainingConfig, null, tokens);

            // Assert
            Assert.That(metrics, Is.Not.Null);
            Assert.That(metrics.Perplexity, Is.GreaterThan(0));
            Assert.That(float.IsFinite((float)metrics.Perplexity));
            Assert.That(metrics.NGramCount, Is.EqualTo(11));
            Assert.That(metrics.CurrentEpoch, Is.EqualTo(1));
        }

        [Test]
        public void BatchingAndTraining_Trigram_EpochsCompare()
        {
            // Arrange
            int[] tokens = new int[] { 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1 };
            ILanguageModel model = new TrigramModel(3);

            var trainingConfig1 = new TrainingConfig(1, 0, 1);
            var trainingConfig2 = new TrainingConfig(20, 0, 5);
            TrainingLoop trainingLoop = new TrainingLoop();

            // Act
            TrainingMetrics metrics1 = trainingLoop.Train(model, null, trainingConfig1, null, tokens);
            TrainingMetrics metrics2 = trainingLoop.Train(model, null, trainingConfig2, null, tokens);

            // Assert
            Assert.That(metrics1, Is.Not.Null);
            Assert.That(metrics2, Is.Not.Null);
            Assert.That(metrics1.Perplexity, Is.EqualTo(metrics2.Perplexity));
            Assert.That(metrics1.NGramCount, Is.EqualTo(metrics2.NGramCount));
            Assert.That(metrics1.CurrentEpoch, Is.EqualTo(1));
            Assert.That(metrics2.CurrentEpoch, Is.EqualTo(20));
        }

        [Test]
        public void BatchingAndTraining_NGram_InvalidToken()
        {
            // Arrange
            int[] tokens = new int[] { 1, 2, 3, 2, -3, 1, 2, 3, 1, 2, 3, 2, 1 };
            ILanguageModel model = new NGramModel(3);

            var trainingConfig = new TrainingConfig(20, 0, 5);
            TrainingLoop trainingLoop = new TrainingLoop();

            // Act + Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => trainingLoop.Train(model, null, trainingConfig, null, tokens));
        }
        [Test]
public void BatchingAndTraining_SeededRng_ProducesSameBatches()
{
    var tokens = new[] { 0, 1, 2, 1, 0, 1, 2 };
    var seed = 42;

    var provider1 = new TokenBatchProvider(tokens);
    var model1 = new NGramModel();
    var config1 = new TrainingConfig(1, 2, 2, 5, null!, seed);
    var trainingLoop1 = new TrainingLoop();
    var metrics1 = trainingLoop1.Train(model1, provider1, config1);

    var provider2 = new TokenBatchProvider(tokens);
    var model2 = new NGramModel();
    var config2 = new TrainingConfig(1, 2, 2, 5, null!, seed);
    var trainingLoop2 = new TrainingLoop();
    var metrics2 = trainingLoop2.Train(model2, provider2, config2);

    Assert.That(metrics1.Perplexity, Is.EqualTo(metrics2.Perplexity), 
        "При однаковому Seed результати навчання мають повністю збігатися.");
    
    Assert.That(metrics1.NGramCount, Is.EqualTo(metrics2.NGramCount), 
        "Кількість оброблених N-грам має бути однаковою.");
        }
[Test]
        public void BatchingAndTraining_ShortTokenStream_Handled()
        {
            int[] tokens = new int[] { 0, 1 };
            ILanguageModel model = new NGramModel(3);

            var trainingConfig = new TrainingConfig(1, 0, 1);
            TrainingLoop trainingLoop = new TrainingLoop();

            Assert.DoesNotThrow(() =>
            {
                trainingLoop.Train(model, null, trainingConfig, null, tokens);
            });
        }
    }
}
