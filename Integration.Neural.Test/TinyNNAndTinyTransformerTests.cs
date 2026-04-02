using Contracts;
using Lib.Models.TinyNN;
using Lib.Models.TinyNN.Factories;
using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;
using Lib.Models.TinyTransformer;
using Lib.Models.TinyTransformer.Factories;
using Lib.MathCore;
using System.Text.Json;

namespace Integration.Neural.Test
{
    public class TinyNNAndTinyTransformerTests
    {
        private int _vocabSize;
        private int[] _tokens;

        private TinyNNConfig _tinyNNConfig;
        private TinyNNWeights _tinyNNWeights;

        [SetUp]
        public void Setup()
        {
            _vocabSize = 10;
            _tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

            _tinyNNConfig = new TinyNNConfig(_vocabSize);
            _tinyNNWeights = new TinyNNWeights(_tinyNNConfig.VocabSize, _tinyNNConfig.EmbeddingSize);
        }

        [Test]
        public void TinyNNAndTinyTransformer_BothImplementILanguageModel()
        {
            var nn = new TinyNNModel("TinyNN", _vocabSize, _tinyNNConfig, _tinyNNWeights);
            var tfFactory = new TinyTransformerModelFactory();
            var tf = tfFactory.Create(_vocabSize);

            Assert.That(nn, Is.InstanceOf<ILanguageModel>());
            Assert.That(tf, Is.InstanceOf<ILanguageModel>());

            Assert.That(nn.NextTokenScores(_tokens).Length, Is.EqualTo(_vocabSize));
            Assert.That(tf.NextTokenScores(ReadOnlySpan<int>.Empty).Length, Is.EqualTo(_vocabSize));
        }

        [Test]
        public void TinyNN_SoftmaxSumsToOne()
        {
            var factory = new TinyNNModelFactory();
            var model = factory.CreateNewModel("tinynn", _vocabSize);

            float[] logits = model.NextTokenScores(_tokens);
            float[] probabilities = SoftmaxCalculator.Softmax(logits); 

            float sum = 0;
            foreach (var p in probabilities)
            {
                sum += p;
            }

            Assert.That(sum, Is.EqualTo(1.0f).Within(1e-6f), "Сума ймовірностей після Softmax має дорівнювати 1.0"); 
            Assert.That(probabilities, Is.All.GreaterThanOrEqualTo(0f), "Ймовірності не можуть бути від'ємними");
        }

        [Test]
        public void TinyNN_CheckpointRoundTrip_PreservesModel()
        {
            var factory = new TinyNNModelFactory();
            var originalModel = factory.CreateNewModel("tinynn", _vocabSize, _tinyNNConfig.EmbeddingSize);

            float[] originalScores = originalModel.NextTokenScores(_tokens);

            var payload = originalModel.ToPayload();
            string json = JsonSerializer.Serialize(payload);

            var doc = JsonDocument.Parse(json);
            var restoredModel = factory.CreateFromPayload(doc.RootElement, "tinynn");

            float[] restoredScores = restoredModel.NextTokenScores(_tokens);

            Assert.Multiple(() =>
            {
                Assert.That(restoredScores.Length, Is.EqualTo(originalScores.Length), "Довжина логітів має збігатися з VocabSize."); 

                for (int i = 0; i < originalScores.Length; i++)
                {
                    Assert.That(restoredScores[i], Is.EqualTo(originalScores[i]).Within(1e-6f), $"Логіт на позиції {i} розбігається після завантаження!");
                }
            });
        }

        [Test]
        public void TinyTransformer_CheckpointRoundTrip_PreservesModel()
        {
            var factory = new TinyTransformerModelFactory();
            int headCount = 2;
            
            var originalModel = factory.Create(_vocabSize, _tinyNNConfig.EmbeddingSize, headCount, _tinyNNConfig.ContextSize);

            float[] originalScores = originalModel.NextTokenScores(_tokens);

            var payload = originalModel.ToPayload();
            string json = JsonSerializer.Serialize(payload);
            
            using var doc = JsonDocument.Parse(json);
            var restoredModel = factory.CreateFromPayload(doc.RootElement);

            float[] restoredScores = restoredModel.NextTokenScores(_tokens);

            Assert.That(restoredScores.Length, Is.EqualTo(originalScores.Length), "Кількість логітів має бути рівною VocabSize");

            for (int i = 0; i < originalScores.Length; i++)
            {
                Assert.That(restoredScores[i], Is.EqualTo(originalScores[i]).Within(1e-6f), $"Розбіжність у вагах трансформера на позиції {i}");
            }
        }

        [Test]
        public void TinyTransformer_ContextSize_TruncatesLongContext()
        {
            int contextSize = 4;
            var factory = new TinyTransformerModelFactory();
            var model = factory.Create(_vocabSize, 8, 1, contextSize, seed: 42);

            int[] shortContext = [1, 2, 3];
            int[] exactContext = [1, 2, 3, 4];
            int[] longContext = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

            float[] shortScores = model.NextTokenScores(shortContext);
            float[] exactScores = model.NextTokenScores(exactContext);
            float[] longScores = model.NextTokenScores(longContext);

            Assert.That(shortScores.Length, Is.EqualTo(_vocabSize));
            Assert.That(exactScores.Length, Is.EqualTo(_vocabSize));
            Assert.That(longScores.Length, Is.EqualTo(_vocabSize));

            int[] longContextLastFour = [6, 7, 8, 9];
            float[] expectedScores = model.NextTokenScores(longContextLastFour);

            for (int i = 0; i < _vocabSize; i++)
            {
                Assert.That(longScores[i], Is.EqualTo(expectedScores[i]).Within(1e-6f));
            }
        }

        [Test]
        public void TinyTransformer_SoftmaxSumsToOne()
        {
            var factory = new TinyTransformerModelFactory();
            var model = factory.Create(_vocabSize, seed: 42);

            float[] logits = model.NextTokenScores(_tokens);
            float[] probabilities = MathOps.Default.Softmax(logits);

            float sum = 0;
            foreach (var p in probabilities)
            {
                sum += p;
            }

            Assert.That(sum, Is.EqualTo(1.0f).Within(1e-6f));
            Assert.That(probabilities, Is.All.GreaterThanOrEqualTo(0f));
        }

        [Test]
        public void TinyTransformer_CanGenerateTokenWithMathOps()
        {
            var factory = new TinyTransformerModelFactory();
            var model = factory.Create(_vocabSize, seed: 42);

            float[] logits = model.NextTokenScores(_tokens);

            int greedyToken = MathOps.Default.ArgMax(logits);
            Assert.That(greedyToken, Is.InRange(0, _vocabSize - 1));

            float[] probs = MathOps.Default.Softmax(logits);
            int sampledToken = MathOps.Default.SampleFromProbs(probs, new Random(123));
            Assert.That(sampledToken, Is.InRange(0, _vocabSize - 1));
        }
    }
}