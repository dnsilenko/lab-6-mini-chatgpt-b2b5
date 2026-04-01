namespace Lib.Models.TinyTransformer.State
{
    public class TinyTransformerWeights
    {
        public float[,] TokenEmbeddings { get; set; } = null!;
        public float[,] Wq { get; set; } = null!;
        public float[,] Wk { get; set; } = null!;
        public float[,] Wv { get; set; } = null!;
        public float[,] Wo { get; set; } = null!;
        public float[,] Ffn1 { get; set; } = null!;
        public float[] Ffn1Bias { get; set; } = null!;
        public float[,] Ffn2 { get; set; } = null!;
        public float[] Ffn2Bias { get; set; } = null!;
        public float[,] OutputW { get; set; } = null!;
        public float[] OutputBias { get; set; } = null!;

        public TinyTransformerWeights()
        {
        }

        public static TinyTransformerWeights Initialize(int vocabSize, int embeddingSize, Random? random = null)
        {
            if (random == null)
            {
                random = new Random();
            }
            int dff = 4 * embeddingSize;
            float scale = 0.02f;

            TinyTransformerWeights weights = new TinyTransformerWeights
            {
                TokenEmbeddings = InitMatrix(vocabSize, embeddingSize, scale, random),
                Wq = InitMatrix(embeddingSize, embeddingSize, scale, random),
                Wk = InitMatrix(embeddingSize, embeddingSize, scale, random),
                Wv = InitMatrix(embeddingSize, embeddingSize, scale, random),
                Wo = InitMatrix(embeddingSize, embeddingSize, scale, random),
                Ffn1 = InitMatrix(embeddingSize, dff, scale, random),
                Ffn1Bias = new float[dff],
                Ffn2 = InitMatrix(dff, embeddingSize, scale, random),
                Ffn2Bias = new float[embeddingSize],
                OutputW = InitMatrix(embeddingSize, vocabSize, scale, random),
                OutputBias = new float[vocabSize]
            };

            return weights;
        }

        private static float[,] InitMatrix(int rows, int cols, float scale, Random random)
        {
            float[,] matrix = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = (float)(random.NextDouble() * 2 - 1) * scale;
                }
            }
            return matrix;
        }
    }
}
