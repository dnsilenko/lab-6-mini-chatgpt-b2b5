using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;

namespace Lib.Models.TinyNN.Layers
{
    public class EmbeddingLayer
    {
        public int VocabSize {get; set;}
        private TinyNNConfig _config;
        private TinyNNWeights _weights;

        public EmbeddingLayer(int vocabSize, TinyNNConfig tinyNNConfig, TinyNNWeights tinyNNWeights)
        {
            VocabSize = vocabSize;
            _config = tinyNNConfig;
            _weights = tinyNNWeights;
        }

        public float[] EncodeContext(int[] context)
        {
            if (context.Length == 0)
            {
                throw new ArgumentException("Context cannot be empty!");
            }
            
            context = ContextCutter(context);
            float[] hidden = new float[_config.EmbeddingSize];

            for(int i = 0; i<context.Length; i++)
            {
                float[] vector = GetVectorFromId(context[i]);
                for(int j = 0; j < _config.EmbeddingSize; j++)
                {
                    hidden[j] += vector[j];
                }
            }

            for (int i = 0; i < _config.EmbeddingSize; i++)
            {
                hidden[i] = hidden[i] / context.Length;
            }

            return hidden;
        }

        public float[] GetVectorFromId(int id)
        {
            if (id < 0 || id >= _config.VocabSize)
            {
                throw new ArgumentOutOfRangeException($"Token ID {id} is out of vocabulary range (0-{_config.VocabSize - 1})!");
            }
            return _weights.Embeddings[id];
        }

        public int[] ContextCutter(int[] context)
        {
            if (context.Length <= _config.ContextSize)
            {
                return context;
            }

            int[] newContext = new int[_config.ContextSize];
            for (int i = 0; i < _config.ContextSize; i++)
            {
                newContext[i] = context[context.Length-_config.ContextSize+i];
            }

            return newContext;
        }  
    }
}