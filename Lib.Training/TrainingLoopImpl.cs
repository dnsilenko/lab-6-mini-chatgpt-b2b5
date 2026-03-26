using Contracts;
using Lib.Models.TinyNN;
using Lib.Training.Configuration;
using Lib.Training.Metrics;
using Lib.Training.Scheduling;
using System.Diagnostics.Metrics;
using System.Text.Json;

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

            metrics.UpdateTinyNN(i + 1, averageLoss, counter, delta);
        }

        return metrics;
    }    

    public TrainingMetrics TrainNGram(ILanguageModel model, IBatchProvider batchProvider, TrainingConfig config)
    {
        int n;
        INGramModels nGramModel;

        if (model.ModelKind == "bigram" && model is NGramModel bigramModel)
        {
            nGramModel = bigramModel;
            n = 2;
        }
        else if (model.ModelKind == "trigram" && model is TrigramModel trigramModel)
        {
            nGramModel = trigramModel;
            n = 3;
        }
        else
        {
            throw new InvalidCastException("Invalid model");
        }

        TrainingMetrics metrics = new TrainingMetrics();
        int[] tokens = batchProvider.GetBatch();

        for (int i = 0; i < config.Epochs; i++)
        {
            if (i == 0 || i == config.Epochs - 1)
            {
                float perplexity = 0;
                int nGramCount = 0;

                DateTime startTime = DateTime.Now;

                nGramModel.Train(tokens);

                DateTime finishTime = DateTime.Now;

                TimeSpan delta = finishTime - startTime;

                PerplexityCalculator calculator = new PerplexityCalculator();

                if (n == 2)
                {
                    perplexity = calculator.ComputePerplexityBigram((NGramModel)nGramModel, tokens);                    
                }
                else
                {
                    perplexity = calculator.ComputePerplexityTrigram((TrigramModel)nGramModel, tokens);
                }

                nGramCount = tokens.Length - n + 1;

                metrics.UpdateNGram(i + 1, perplexity, nGramCount, delta);

                if (CheckpointScheduler.ScheduleCheck(i + 1, config.CheckpointInterval, config.Epochs))
                {
                    var json = model.GetPayloadForCheckpoint();
                }
            }           
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