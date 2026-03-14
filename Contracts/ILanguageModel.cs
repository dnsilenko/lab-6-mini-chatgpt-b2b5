namespace Contracts;

public interface ILanguageModel
{
    string ModelKind { get; } // "ngram", "trigram", "tinynn", "tinytransformer"
    int VocabSize { get; }
    float[] NextTokenScores(ReadOnlySpan<int> context); // логiти для кожного токена
    object GetPayloadForCheckpoint();
}