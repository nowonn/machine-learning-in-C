/* date = December 16th 2023 10:50 am */

#ifndef NOWONMLLIB_H
#define NOWONMLLIB_H

typedef enum { POLYNOMIAL, LOGISTIC } ModelType;

typedef struct {
    ModelType type;
    double **xTrain;
    double *yTrain;
    int setSize;
    double *parameters;
    int parameterAmount;
    double bias;
    double *featuresMean;
    double *featuresStandardDeviation;
    double regularizationParameter;
} Model;

typedef struct {
    double *m;
    double *v;
    double mBias;
    double vBias;
    int t;
} AdamOptimizer;

double ComputeFeatureMean(double **xTrain, Model *model, int featureIndex);

double ComputeFeatureStandardDeviation(double **xTrain, Model *model, int featureIndex);

double **ZScoreNormalize(double **dataX, Model *model);

Model *CreateRegressionModel(ModelType type, double regularizationParameter);

double Sigmoid(double x);

double ComputeY(Model *model, double *input);

double MeanSquaredErrorLoss(Model *model, double guess, double expectedOutput);

double LogisticLoss(Model *model, double guess, double expectedOutput);

double ComputeRegularizationTerm(Model *model);

double ComputeCost(Model *model);

double ComputeGradient(Model *model, int parameterIndex, int isBias);

void GradientDescent(Model *model, double learningRate);

AdamOptimizer *CreateAdamOptimizer(int parameterAmount);

void AdamOptimization(Model *model, AdamOptimizer *optimizer, double learningRate, double beta1, double beta2, double epsilon);

void Train(Model *model, double **dataX, double *dataY, int setSize, int parameterAmount, long iterations, double learningRate, int polynomialOrder);

void PredictOutput(Model *model, double **input, int inputSize, double *output);

#endif //NOWONMLLIB_H
