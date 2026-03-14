namespace Contracts;

public interface ITokenizer
{
    int VocabSize { get; }
    int[] Encode(string text); // текст → масив ID токенiв
    string Decode(ReadOnlySpan<int> tokens); // ID → текст
    object GetPayloadForCheckpoint(); // для збереження в чекпоiнт
}
