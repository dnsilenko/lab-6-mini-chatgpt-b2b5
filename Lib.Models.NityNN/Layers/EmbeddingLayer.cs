namespace Lib.Models.TinyNN.Layers
{
    internal class EmbeddingLayer
    {
        public const int Embedding_Size = 32; // Ці константи, ми повинні брати із конфігу
        public const int Context_Size = 8;
        public float[] EncodeContext(int[] context, float[,] embeddings)
        {
            context = ContextCutter(context);
            float[] hidden = new float[Embedding_Size];

            for(int i = 0; i<context.Length; i++)
            {
                float[] vector = GetVectorFromId(context[i], embeddings);
                for(int j = 0; j < Embedding_Size; j++)
                {
                    hidden[j] += vector[j];
                }
            }

            for (int i = 0; i < Embedding_Size; i++)
            {
                hidden[i] = hidden[i] / context.Length;
            }

            return hidden;
        }

        public float[] GetVectorFromId(int id, float[,] embeddings)
        {
            float[] vector = new float[Embedding_Size];
            for (int i = 0; i < Embedding_Size; i++)
            {
                vector[i] = embeddings[id, i];
            }
            return vector;
        }

        public int[] ContextCutter(int[] context)
        {
            if (context.Length <= Context_Size)
            {
                return context;
            }

            int[] newContext = new int[Context_Size];
            for (int i = 0; i < Context_Size; i++)
            {
                newContext[i] = context[context.Length-Context_Size+i];
            }

            return newContext;
        }  
    }
}
