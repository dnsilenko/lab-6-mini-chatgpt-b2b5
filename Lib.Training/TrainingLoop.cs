using Contracts;
using Lib.Training.Configuration;
using Lib.Training.Metrics;

namespace Lib.Training;

public class TrainingLoop : ITrainingLoop
{
    public TrainingMetrics Train (ILanguageModel model, IBatchProvider batchProvider, TrainingConfig config)
    {
        TrainingLoopImpl loopImpl = new TrainingLoopImpl();

        if (model.ModelKind == "NGram")
        {
            throw new NotImplementedException(); // implement this case, please :)
        }
        else if (model.ModelKind == "TinyNN")
        {
            return loopImpl.TrainTinyNN(model, batchProvider, config);
        }                        
        else if (model.ModelKind == "Transformer")
        {
            new TrainingMetrics();
        }

        throw new ArgumentException("Invalid data");
    }
}