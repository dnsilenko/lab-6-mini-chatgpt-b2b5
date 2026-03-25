namespace Lib.Training.Metrics
{
    public class TrainingMetrics
    {
        public int CurrentEpoch { get; private set; }
        public float AverageLoss { get; private set; }
        public int TotalSteps { get; private set; }
        public TimeSpan ElapsedTime { get; private set; }

        public TrainingMetrics() {}

        public void Update(int currentEpoch, float averageLoss, int totalSteps, TimeSpan elapsedTime)
        {
            if(currentEpoch < 0 || averageLoss < 0 || totalSteps < 0 || elapsedTime < TimeSpan.Zero)
            {
                throw new ArgumentException("Invalid training metrics values!");
            }

            CurrentEpoch = currentEpoch;
            AverageLoss = averageLoss;
            TotalSteps = totalSteps;
            ElapsedTime = elapsedTime;

        }
    }
}