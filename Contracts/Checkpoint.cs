namespace Contracts;

public record Checkpoint(
    string ModelKind, // "ngram", "trigram", "tinynn", "tinytransformer"
    string TokenizerKind, // "word" або "char"
    object TokenizerPayload, // словник/словник слiв
    object ModelPayload, // ваги/ймовiрностi моделi
    int Seed,
    string ContractFingerprintChain
);