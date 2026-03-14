using System;
using System.Collections.Generic;
using System.Text;

namespace Lib.Models.TinyNN.Layers
{
    internal class LinearHead
    {
        public float[] Project(float[] hidden, float[,] outputweights, float[] outputbias)
        {
           float[] vector = MultiplyHiddenOnWeights(hidden, outputweights);
           float[] logits = AddBiasToVector(vector, outputbias);
           return logits; 
        }

        public float[] AddBiasToVector(float[] vector, float[] outputbias)
        {
            for (int i = 0; i < outputbias.Length; i++)
            {
                vector[i] += outputbias[i];
            }

            return vector;
        }

        public float[] MultiplyHiddenOnWeights(float[] hidden, float[,] outputweights)
        {
            float[] vector = new float[outputweights.GetLength(1)];
            for(int i = 0; i < outputweights.GetLength(1); i++)
            {
                for (int j = 0; j < hidden.Length; j++)
                {
                    vector[i] += hidden[j] * outputweights[j, i];
                }
            }
            return vector;
        }
    }
}
