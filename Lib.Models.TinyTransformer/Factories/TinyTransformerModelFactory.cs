using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer.Factories
{
    public class TinyTransformerModelFactory
    {
        public TinyTransformerModel Create(TinyTransformerConfig config, TinyTransformerWeights weights)
        {
            return new TinyTransformerModel(config, weights);
        }

        public TinyTransformerModel Create(int vocabSize, int? seed = null)
        {
            TinyTransformerConfig config = new TinyTransformerConfig(vocabSize);
            Random random = seed.HasValue ? new Random(seed.Value) : new Random();
            TinyTransformerWeights weights = TinyTransformerWeights.Initialize(vocabSize, config.EmbeddingSize, random);
            return new TinyTransformerModel(config, weights);
        }

        public TinyTransformerModel Create(int vocabSize, int embeddingSize, int headCount, int contextSize, int? seed = null)
        {
            TinyTransformerConfig config = new TinyTransformerConfig(vocabSize, embeddingSize, headCount, contextSize);
            Random random = seed.HasValue ? new Random(seed.Value) : new Random();
            TinyTransformerWeights weights = TinyTransformerWeights.Initialize(vocabSize, embeddingSize, random);
            return new TinyTransformerModel(config, weights);
        }
    }
}
