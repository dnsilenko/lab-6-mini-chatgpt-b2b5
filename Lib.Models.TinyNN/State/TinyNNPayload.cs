using Lib.Models.TinyNN.Configuration;
namespace Lib.Models.TinyNN.State
{
    internal class TinyNNPayload
    {
        public TinyNNConfig Config {get; set;}
        public float[][] Embeddings {get; set;}
        public float[][] OutputWeights {get; set;}
        public float[] OutputBias {get; set;}
    }
}