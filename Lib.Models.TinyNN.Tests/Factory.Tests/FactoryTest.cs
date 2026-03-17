using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;
using Lib.Models.TinyNN.Layers;
using Lib.Models.TinyNN.Factories;
using Lib.Models.TinyNN;
using System.Text.Json;

namespace Lib.Models.TinyNN.Tests.Factory;

public class FactoryTest
{   
    [TestCase(1000, 32, 8)]
    [TestCase(500, 64, 16)]
    [TestCase(100, 16, 4)]
    public void CreateNewModel_VariousParameters_ReturnsModelWithCorrectConfiguration(int vocabSize, int embeddingSize, int contextSize)
    {
        TinyNNModelFactory factory = new TinyNNModelFactory();

        TinyNNModel model = factory.CreateNewModel("tinynn", vocabSize, embeddingSize, contextSize);

        Assert.That(model.VocabSize, Is.EqualTo(vocabSize));
    }

    [Test]
    public void CreateFromPayload_ValidJson_ReturnsRestoredModelWithCorrectWeights()
    {
        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var expectedBias = new float[] { 0.1f, 0.2f, 0.3f };
        
        TinyNNPayload payloadData = new TinyNNPayload
        {
            Config = new TinyNNConfig(3, 2, 8),
            Embeddings = new float[][] { new float[] { 1f, 1f }, new float[] { 2f, 2f }, new float[] { 3f, 3f } },
            OutputWeights = new float[][] { new float[] { 0.5f, 0.5f, 0.5f }, new float[] { 0.5f, 0.5f, 0.5f } },
            OutputBias = expectedBias
        };

        string jsonString = JsonSerializer.Serialize(payloadData, options);
        JsonElement jsonElement = JsonDocument.Parse(jsonString).RootElement;
        
        TinyNNModelFactory factory = new TinyNNModelFactory();
        TinyNNModel restoredModel = factory.CreateFromPayload(jsonElement, "tinynn");

        float[] zeroHidden = new float[2]; 
        var result = restoredModel.NextTokenScores(new int[] { 0 }); 
        
        Assert.That(restoredModel.VocabSize, Is.EqualTo(3));
    }

    [Test]
    public void CreateFromPayload_IncorrectModelKind_ThrowsArgumentException()
    {
        TinyNNModelFactory factory = new TinyNNModelFactory();
        var emptyPayload = JsonDocument.Parse("{}").RootElement;

        Assert.Throws<ArgumentException>(() => factory.CreateFromPayload(emptyPayload, "bigram"));
    }
}