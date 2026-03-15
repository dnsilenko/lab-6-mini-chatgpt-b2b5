using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;
using Lib.Models.TinyNN.Layers;

namespace Layers.Tests;

public class EmbeddingLayerTest
{
    private int _vocabSize = 10;
    private TinyNNConfig _config;
    private TinyNNWeights _weights;
    private EmbeddingLayer _layer;

    [SetUp]
    public void SetUp()
    {
        _config = new TinyNNConfig(_vocabSize);
        _weights = new TinyNNWeights(_vocabSize, _config.EmbeddingSize);
        _layer = new EmbeddingLayer(_vocabSize, _config, _weights); 
    }

    [Test]
    public void CheckLengthOfHiddenTest1()
    {
        int[] context = new int[] {0, 3, 5, 2, 7};
        float[] hidden = _layer.EncodeContext(context);
        Assert.That(hidden.Length, Is.EqualTo(32));
    }

    [Test]
    public void CheckLengthOfHiddenTest2()
    {
        int[] context = new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        float[] hidden = _layer.EncodeContext(context);
        Assert.That(hidden.Length, Is.EqualTo(32));
    }

    [Test]
    public void IfContextIsEmptyTest()
    {
        int[] context = new int[] {};
        Assert.Throws<ArgumentException>(() => _layer.EncodeContext(context));
    }

    [Test]
    public void InvalidIdTest()
    {
        int invalidId = 11;
        Assert.Throws<ArgumentOutOfRangeException>(() => _layer.GetVectorFromId(invalidId));
    }

    [Test]
    public void ContextCutterTest()
    {
        int[] context1 = new int[] {2, 3, 4, 5, 6, 7, 8, 9};
        int[] context2 = new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        int[] cuttedContext1 = _layer.ContextCutter(context1);
        int[] cuttedContext2 = _layer.ContextCutter(context2);

        Assert.That(cuttedContext1, Is.EqualTo(cuttedContext2));
    }
}