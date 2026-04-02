using Lib.Models.TinyNN;
using Lib.Models.TinyNN.Factories;
using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;
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
            var tinyNN = new TinyNNModel("TinyNN", _vocabSize, _tinyNNConfig, _tinyNNWeights);

            float[] logits = tinyNN.NextTokenScores(_tokens);

            Assert.That(logits.Length, Is.EqualTo(_vocabSize));
            for (int i = 0; i < logits.Length; i++)
            {
                Assert.That(logits[i], Is.Not.NaN);
            }
        }

        [Test]
        public void TinyNN_SoftmaxSumsToOne()
        {
            var factory = new TinyNNModelFactory();
            int vocabSize = 50;
            var model = factory.CreateNewModel("tinynn", vocabSize);

            var context = new int[] { 1, 2, 3 };

            float[] logits = model.NextTokenScores(context);
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
            int vocabSize = 100;
            int embeddingSize = 32;
            var originalModel = factory.CreateNewModel("tinynn", vocabSize, embeddingSize);

            var testContext = new int[] { 1, 5, 10, 42 };

            float[] originalScores = originalModel.NextTokenScores(testContext);

            var payload = originalModel.ToPayload();
            string json = JsonSerializer.Serialize(payload);

            var doc = JsonDocument.Parse(json);
            var restoredModel = factory.CreateFromPayload(doc.RootElement, "tinynn");

            float[] restoredScores = restoredModel.NextTokenScores(testContext);

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
            int vocabSize = 20;
            int embeddingSize = 32;
            int headCount = 2;
            int contextSize = 8192;
            
            var originalModel = factory.Create(vocabSize, embeddingSize, headCount, contextSize);
            
            var testContext = new int[] { 1, 2, 3 };

            float[] originalScores = originalModel.NextTokenScores(testContext);

            var payload = originalModel.ToPayload();
            string json = JsonSerializer.Serialize(payload);
            
            using var doc = JsonDocument.Parse(json);
            var restoredModel = factory.CreateFromPayload(doc.RootElement);

            float[] restoredScores = restoredModel.NextTokenScores(testContext);

            Assert.That(restoredScores.Length, Is.EqualTo(originalScores.Length), "Кількість логітів має бути рівною VocabSize");

            for (int i = 0; i < originalScores.Length; i++)
            {
                Assert.That(restoredScores[i], Is.EqualTo(originalScores[i]).Within(1e-6f), $"Розбіжність у вагах трансформера на позиції {i}");
            }
        }  
    }
}