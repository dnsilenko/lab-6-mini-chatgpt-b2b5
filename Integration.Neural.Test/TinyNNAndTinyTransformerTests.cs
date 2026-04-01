using Lib.Models.TinyNN;
using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;

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
    }
}