using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Layers;
using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer
{
    public class TinyTransformerModel
    {
        private readonly TinyTransformerConfig _config;
        private readonly TinyTransformerWeights _weights;
        private readonly SelfAttentionLayer _attention;

        public string ModelKind => "tinytransformer";
        public int VocabSize => _config.VocabSize;
        public int ContextSize => _config.ContextSize;

        public TinyTransformerModel(TinyTransformerConfig config, TinyTransformerWeights weights)
        {
            _config = config;
            _weights = weights;
            _attention = new SelfAttentionLayer();
        }

        public float[] Compute(ReadOnlySpan<int> context, int embeddingSize)
        {
            int n = Math.Min(context.Length, _config.ContextSize);
            int start = context.Length > _config.ContextSize ? context.Length - _config.ContextSize : 0;

            float[][] embeddings = new float[n][];
            for (int i = 0; i < n; i++)
            {
                int tokenId = context[start + i];
                embeddings[i] = new float[embeddingSize];
                for (int j = 0; j < embeddingSize; j++)
                {
                    embeddings[i][j] = _weights.TokenEmbeddings[tokenId, j];
                }
            }

            float[][] attnOutput = _attention.Compute(embeddings, _weights, embeddingSize);

            return attnOutput[n - 1];
        }

        public float[] Project(float[] hidden, int vocabSize)
        {
            int d = hidden.Length;
            int dff = 4 * d;

            float[] ffnHidden = new float[dff];
            for (int j = 0; j < dff; j++)
            {
                float sum = _weights.Ffn1Bias[j];
                for (int i = 0; i < d; i++)
                {
                    sum += hidden[i] * _weights.Ffn1[i, j];
                }
                ffnHidden[j] = Math.Max(0, sum);
            }

            float[] ffnOut = new float[d];
            for (int j = 0; j < d; j++)
            {
                float sum = _weights.Ffn2Bias[j];
                for (int i = 0; i < dff; i++)
                {
                    sum += ffnHidden[i] * _weights.Ffn2[i, j];
                }
                ffnOut[j] = sum;
            }

            float[] logits = new float[vocabSize];
            for (int j = 0; j < vocabSize; j++)
            {
                float sum = _weights.OutputBias[j];
                for (int i = 0; i < d; i++)
                {
                    sum += ffnOut[i] * _weights.OutputW[i, j];
                }
                logits[j] = sum;
            }

            return logits;
        }

        public float[] Forward(ReadOnlySpan<int> context, int vocabSize, int embeddingSize)
        {
            float[] hidden = Compute(context, embeddingSize);
            return Project(hidden, vocabSize);
        }

        public float[] NextTokenScores(ReadOnlySpan<int> context)
        {
            if (context.Length == 0)
            {
                return new float[_config.VocabSize];
            }

            return Forward(context, _config.VocabSize, _config.EmbeddingSize);
        }

        public TinyTransformerPayload ToPayload()
        {
            return new TinyTransformerPayload
            {
                Config = this._config,
                TokenEmbeddings = ToJaggedArray(this._weights.TokenEmbeddings),
                Wq = ToJaggedArray(this._weights.Wq),
                Wk = ToJaggedArray(this._weights.Wk),
                Wv = ToJaggedArray(this._weights.Wv),
                Wo = ToJaggedArray(this._weights.Wo),
                Ffn1 = ToJaggedArray(this._weights.Ffn1),
                Ffn1Bias = this._weights.Ffn1Bias,
                Ffn2 = ToJaggedArray(this._weights.Ffn2),
                Ffn2Bias = this._weights.Ffn2Bias,
                OutputW = ToJaggedArray(this._weights.OutputW),
                OutputBias = this._weights.OutputBias
            };
        }

        public object GetPayloadForCheckpoint()
        {
            return new
            {
                config = new
                {
                    vocabSize = _config.VocabSize,
                    embeddingSize = _config.EmbeddingSize,
                    headCount = _config.HeadCount,
                    contextSize = _config.ContextSize
                },
                tokenEmbeddings = ToJaggedArray(_weights.TokenEmbeddings),
                wq = ToJaggedArray(_weights.Wq),
                wk = ToJaggedArray(_weights.Wk),
                wv = ToJaggedArray(_weights.Wv),
                wo = ToJaggedArray(_weights.Wo),
                ffn1 = ToJaggedArray(_weights.Ffn1),
                ffn1Bias = _weights.Ffn1Bias,
                ffn2 = ToJaggedArray(_weights.Ffn2),
                ffn2Bias = _weights.Ffn2Bias,
                outputW = ToJaggedArray(_weights.OutputW),
                outputBias = _weights.OutputBias
            };
        }

        private static float[][] ToJaggedArray(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            float[][] result = new float[rows][];
            for (int i = 0; i < rows; i++)
            {
                result[i] = new float[cols];
                for (int j = 0; j < cols; j++)
                {
                    result[i][j] = matrix[i, j];
                }
            }
            return result;
        }
    }
}
