using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;

namespace Lib.Models.TinyNN.Layers
{
    public class LinearHead
    {
        public int VocabSize {get; set;}
        private TinyNNConfig _config;
        private TinyNNWeights _weights;

        public LinearHead(int vocabSize, TinyNNConfig tinyNNConfig, TinyNNWeights tinyNNWeights)
        {
            VocabSize = vocabSize;
            _config = tinyNNConfig;
            _weights = tinyNNWeights;
        }

        public float[] Project(float[] hidden)
        {
           float[] vector = MultiplyHiddenOnWeights(hidden);
           float[] logits = AddBiasToVector(vector);
           return logits; 
        }

        public float[] AddBiasToVector(float[] vector)
        {
            for (int i = 0; i < _weights.OutputBias.Length; i++)
            {
                vector[i] += _weights.OutputBias[i];
            }

            return vector;
        }

        public float[] MultiplyHiddenOnWeights(float[] hidden)
        {
            float[] vector = new float[_config.VocabSize];
            for(int i = 0; i < _config.VocabSize; i++)
            {
                for (int j = 0; j < hidden.Length; j++)
                {
                    vector[i] += hidden[j] * _weights.OutputWeights[j][i];
                }
            }
            return vector;
        }
    }
}