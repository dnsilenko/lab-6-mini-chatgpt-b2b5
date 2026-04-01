using System;

namespace Lib.Models.TinyTransformer.Layers
{
    public class SelfAttentionLayer
    {
        public float[][] Compute(float[][] x, dynamic weights, int d)
        {
            int n = x.Length;
            float[][] Q = Multiply(x, weights.Wq, d);
            float[][] K = Multiply(x, weights.Wk, d);
            float[][] V = Multiply(x, weights.Wv, d);

            float[][] scores = new float[n][];
            float scale = (float)Math.Sqrt(d);

            for (int i = 0; i < n; i++)
            {
                scores[i] = new float[n];
                for (int j = 0; j < n; j++)
                {
                    if (j > i)
                    {
                        scores[i][j] = float.NegativeInfinity;
                        continue;
                    }

                    float dot = 0;
                    for (int k = 0; k < d; k++)
                    {
                        dot += Q[i][k] * K[j][k];
                    }
                    scores[i][j] = dot / scale;
                }
            }

            float[][] output = new float[n][];
            for (int i = 0; i < n; i++)
            {
                float[] attentionWeights = Softmax(scores[i]);
                output[i] = new float[d];
                for (int j = 0; j <= i; j++)
                {
                    for (int k = 0; k < d; k++)
                    {
                        output[i][k] += attentionWeights[j] * V[j][k];
                    }
                }
            }

            return Multiply(output, weights.Wo, d);
        }

        private float[][] Multiply(float[][] input, float[,] matrix, int d)
        {
            int n = input.Length;
            float[][] result = new float[n][];
            for (int i = 0; i < n; i++)
            {
                result[i] = new float[d];
                for (int j = 0; j < d; j++)
                {
                    for (int k = 0; k < d; k++)
                    {
                        result[i][j] += input[i][k] * matrix[k, j];
                    }
                }
            }
            return result;
        }

        private float[] Softmax(float[] logits)
        {
            float max = float.NegativeInfinity;
            foreach (float v in logits)
            {
                if (v > max)
                {
                    max = v;
                }
            }

            float[] exp = new float[logits.Length];
            float sum = 0;
            for (int i = 0; i < logits.Length; i++)
            {
                if (float.IsNegativeInfinity(logits[i]))
                {
                    exp[i] = 0;
                }
                else
                {
                    exp[i] = (float)Math.Exp(logits[i] - max);
                    sum += exp[i];
                }
            }
            for (int i = 0; i < exp.Length; i++)
            {
                exp[i] /= sum;
            }
            return exp;
        }
    }
}
