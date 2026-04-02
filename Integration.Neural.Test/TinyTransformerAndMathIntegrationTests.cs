using System;
using System.Linq;
using NUnit.Framework;
using Lib.MathCore;
using Lib.Models.TinyNN;
using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;
using Lib.Models.TinyTransformer;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.State;

namespace Integration.Neural.Test
{
    [TestFixture]
    public class TinyTransformerAndMathIntegrationTests
    {
        private float[][] Create2DArray(int rows, int cols)
        {
            float[][] arr = new float[rows][];
            for (int i = 0; i < rows; i++) arr[i] = new float[cols];
            return arr;
        }

        private TinyTransformerWeights CreateDummyWeights(int vocabSize, int embeddingSize, int contextSize)
        {
            return new TinyTransformerWeights(
                Create2DArray(vocabSize, embeddingSize),
                Create2DArray(contextSize, embeddingSize),
                Create2DArray(embeddingSize, embeddingSize),
                Create2DArray(embeddingSize, embeddingSize),
                Create2DArray(embeddingSize, embeddingSize),
                Create2DArray(embeddingSize, embeddingSize),
                new float[embeddingSize],
                Create2DArray(embeddingSize, embeddingSize * 4),
                new float[embeddingSize * 4],
                Create2DArray(embeddingSize * 4, embeddingSize),
                new float[embeddingSize]
            );
        }

        [Test]
        public void TinyTransformer_ContextSize_TruncatesLongContext()
        {
            int vocabSize = 10;
            int embeddingSize = 16;
            int headCount = 1;
            int contextSize = 4;

            var config = new TinyTransformerConfig(vocabSize, embeddingSize, headCount, contextSize);
            var weights = CreateDummyWeights(vocabSize, embeddingSize, contextSize);
            var transformer = new TinyTransformerModel(config, weights);

            int[] longContext = { 1, 2, 3, 4, 5, 6, 7 };

            Assert.DoesNotThrow(() =>
            {
                float[] logits = transformer.NextTokenScores(longContext);
                Assert.That(logits.Length, Is.EqualTo(vocabSize));
            });
        }

        [Test]
        public void ModelsLogits_WithSoftmaxCalculator_SumsToOne()
        {
            int vocabSize = 10;
            int[] context = { 1, 2, 3 };

            var transformerConfig = new TinyTransformerConfig(vocabSize, 16, 1, 8);
            var transformerWeights = CreateDummyWeights(vocabSize, 16, 8);
            var transformer = new TinyTransformerModel(transformerConfig, transformerWeights);

            var nnConfig = new TinyNNConfig(vocabSize);
            var nnWeights = new TinyNNWeights(nnConfig.VocabSize, nnConfig.EmbeddingSize);
            var nn = new TinyNNModel("TinyNN", vocabSize, nnConfig, nnWeights);

            float[] transformerLogits = transformer.NextTokenScores(context);
            float[] transformerProbs = SoftmaxCalculator.Softmax(transformerLogits);

            float[] nnLogits = nn.NextTokenScores(context);
            float[] nnProbs = SoftmaxCalculator.Softmax(nnLogits);

            float transformerSum = transformerProbs.Sum();
            float nnSum = nnProbs.Sum();

            Assert.That(transformerSum, Is.EqualTo(1.0f).Within(0.001f));
            Assert.That(nnSum, Is.EqualTo(1.0f).Within(0.001f));
        }
    }
}