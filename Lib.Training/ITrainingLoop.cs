using Contracts;
using Lib.Training.Configuration;
using Lib.Training.Metrics;

namespace Lib.Training;

public interface ITrainingLoop
{
    TrainingMetrics Train(ILanguageModel model, IBatchProvider batchProvider, TrainingConfig config);
}

public interface IBatchProvider
{
    int[] GetBatch();
}   