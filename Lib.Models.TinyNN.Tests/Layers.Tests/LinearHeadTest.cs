using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;
using Lib.Models.TinyNN.Layers;
using NUnit.Framework;

namespace Layers.Tests;

public class LinearHeadTest
{
    private int _vocabSize = 10;
    private TinyNNConfig _config;
    private TinyNNWeights _weights;
    private EmbeddingLayer _embeddinglayer;
    private LinearHead _linearhead;

    [SetUp]
    public void SetUp()
    {
        _config = new TinyNNConfig(_vocabSize);
        _weights = new TinyNNWeights(_vocabSize, _config.EmbeddingSize);
        _embeddinglayer = new EmbeddingLayer(_vocabSize, _config, _weights); 
        _linearhead = new LinearHead(_vocabSize, _config, _weights);
    }

    [Test]
    public void Project_ValidHiddenVector_ReturnsLogitsWithVocabSizeLength()
    {
        int[] context = new int[] {0, 3, 5, 2, 7};
        float[] hidden = _embeddinglayer.EncodeContext(context);
        float[] logits = _linearhead.Project(hidden);

        Assert.That(logits.Length, Is.EqualTo(_vocabSize));
    }
}