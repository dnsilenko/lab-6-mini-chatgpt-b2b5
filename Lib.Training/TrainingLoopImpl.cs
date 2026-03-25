using Contracts;
using Lib.Models.TinyNN;
using Lib.Training.Configuration;
using Lib.Training.Metrics;

namespace Lib.Training;

public class TrainingLoopImpl 
{
    public TrainingMetrics TrainTinyNN (ILanguageModel model, IBatchProvider batchProvider, TrainingConfig config)
    {
        if (model is not TinyNNModel tinyNNModel)
            throw new InvalidCastException("Invalid model");

        TrainingMetrics metrics = new TrainingMetrics();
        int[] tokens = batchProvider.GetBatch();

        for (int i = 0; i < config.Epochs; i++)
        {
            float sumLoss = 0f;
            int counter = 0;
            DateTime startTime = DateTime.Now;

            for (int j = 0; j < tokens.Length - 1; j++)
            {
                int[] context = GetContext(tokens, j);
                int target = tokens[j + 1];

                float loss = tinyNNModel.TrainStep(context, target, config.LearningRate);
                sumLoss += loss;
                counter++;
            }

            DateTime finishTime = DateTime.Now;

            TimeSpan delta = finishTime - startTime;

            float averageLoss = 0f;
            if (counter != 0)
            {
                averageLoss = sumLoss / counter;
            }

            metrics.Update(i + 1, averageLoss, counter, delta);
        }

        return metrics;
    }    

    private int[] GetContext (int[] tokens, int endIndex)
    {
        int[] context = new int[endIndex + 1];
        for (int i = 0; i <= endIndex; i++)
        {
            context[i] = tokens[i];  
        }

        return context;  
    }
}