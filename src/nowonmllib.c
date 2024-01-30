#include <stdlib.h>
#include <math.h>
#include "nowonmllib.h"

double ComputeFeatureMean(double **xTrain, Model *model, int featureIndex){
    double sum = 0;
    int setSize = model->setSize;
    for(int i = 0; i < setSize; i++)
        sum += xTrain[i][featureIndex];
    return (sum / setSize);
}

double ComputeFeatureStandardDeviation(double **xTrain, Model *model, int featureIndex){
    double deviation = 0;
    int setSize = model->setSize;
    for(int i = 0; i < setSize; i++)
        deviation += pow(xTrain[i][featureIndex] - model->featuresMean[featureIndex], 2);
    return sqrt(deviation / setSize);
}

double **ZScoreNormalize(double **dataX, Model *model){
    int setSize = model->setSize;
    int parameterAmount = model->parameterAmount;
    double *means = model->featuresMean;
    double *stdDevs = model->featuresStandardDeviation;
    double **normalizedDataX = malloc(sizeof(double*) * setSize);
    for(int i = 0; i < setSize; i++)
        normalizedDataX[i] = malloc(sizeof(double) * parameterAmount);
    
    for(int i = 0; i < setSize; i++){
        for(int j = 0; j < parameterAmount; j++){
            normalizedDataX[i][j] = (dataX[i][j] - means[j]) / stdDevs[j];
        }
    }
    return normalizedDataX;
}

Model *CreateRegressionModel(ModelType type,
                             double regularizationParameter){
    Model *model = malloc(sizeof(Model));
    model->type = type;
    model->regularizationParameter = regularizationParameter;
    
    return model;
} 

double Sigmoid(double x){
    return 1/(1 + exp(-x));
}

double ComputeY(Model *model, double *input){
    double result = model->bias;
    
    for(int i = 0; i < model->parameterAmount; i++)
        result += model->parameters[i] * input[i];
    
    return (model->type == LOGISTIC) ? Sigmoid(result) : result;
}

double MeanSquaredErrorLoss(Model *model, double guess, double expectedOutput){
    return ((guess - expectedOutput)/2)*(guess - expectedOutput);
}

double LogisticLoss(Model *model, double guess, double expectedOutput){
    return -expectedOutput * log(guess) - (1 - expectedOutput) * log(1 - guess);
}

double ComputeRegularizationTerm(Model *model){
    double totalRegularizationTerm = 0;
    for(int i = 0; i < model->setSize; i++){
        totalRegularizationTerm += model->parameters[i] * model->parameters[i];
    }
    totalRegularizationTerm += model->bias * model->bias;
    return totalRegularizationTerm * model->regularizationParameter / (model->setSize << 1);
}

double ComputeCost(Model *model){
    double **input = model->xTrain;
    double *expectedOutput = model->yTrain;
    int setSize = model->setSize;
    double totalCost = 0;
    double (*LossFunction)(Model*, double, double);
    
    switch(model->type){
        case LOGISTIC:
        LossFunction = LogisticLoss;
        break;
        default:
        LossFunction = MeanSquaredErrorLoss;
    }
    
    for(int i = 0; i < setSize; i++){
        double guess = ComputeY(model, input[i]);
        totalCost += LossFunction(model, guess, expectedOutput[i]);
    }
    
    double cost = totalCost/setSize;
    
    if(model->regularizationParameter > 1e-8) cost += ComputeRegularizationTerm(model);
    
    return cost;
}

double ComputeGradient(Model *model, int parameterIndex, 
                       int isBias){
    double totalGradent = 0;
    if(!isBias){
        for(int i = 0; i < model->setSize; i++){
            totalGradent += (ComputeY(model, model->xTrain[i]) - model->yTrain[i]) * model->xTrain[i][parameterIndex];
        }
        return totalGradent / model->setSize;
    }
    
    for(int i = 0; i < model->setSize; i++){
        totalGradent += (ComputeY(model, model->xTrain[i]) - model->yTrain[i]);
    }
    return totalGradent / model->setSize;
}

void GradientDescent(Model *model, double learningRate){
    double aux[model->parameterAmount];
    
    for(int i = 0; i < model->parameterAmount; i++){
        double derivative = ComputeGradient(model, i, 0);;
        aux[i] = derivative * learningRate; 
    }
    
    double derivative = ComputeGradient(model, 0, 1);
    model->bias -= derivative * learningRate;
    
    for(int i = 0; i < model->parameterAmount; i++){
        model->parameters[i] -= aux[i]; 
    }
}

AdamOptimizer *CreateAdamOptimizer(int parameterAmount){
    AdamOptimizer *optimizer = malloc(sizeof(AdamOptimizer));
    optimizer->m = calloc(parameterAmount, sizeof(double));
    optimizer->v = calloc(parameterAmount, sizeof(double));
    optimizer->t = 0;
    optimizer->mBias = 0;
    optimizer->vBias = 0;
    
    return optimizer;
}

void AdamOptimization(Model *model, AdamOptimizer *optimizer, double learningRate, double beta1, double beta2, double epsilon){
    double aux[model->parameterAmount];
    optimizer->t += 1;
    for(int i = 0; i < model->parameterAmount; i++){
        double g = ComputeGradient(model, i, 0);
        optimizer->m[i] = beta1 * optimizer->m[i] + (1 - beta1) * g;
        optimizer->v[i] = beta2 * optimizer->v[i] + (1 - beta2) * g * g;
        double mHat = optimizer->m[i] / (1 - pow(beta1, optimizer->t));
        double vHat = optimizer->v[i] / (1 - pow(beta2, optimizer->t));
        aux[i] = learningRate * mHat / (sqrt(vHat) + epsilon);
    }
    
    double gBias = ComputeGradient(model, 0, 1);
    optimizer->mBias = beta1 * optimizer->mBias + (1 - beta1) * gBias;
    optimizer->vBias = beta2 * optimizer->vBias + (1 - beta2) * gBias * gBias;
    double mHat_bias = optimizer->mBias / (1 - pow(beta1, optimizer->t));
    double vHat_bias = optimizer->vBias / (1 - pow(beta2, optimizer->t));
    double auxBias = learningRate * mHat_bias / (sqrt(vHat_bias) + epsilon);
    
    model->bias -= auxBias;
    
    for(int i = 0; i < model->parameterAmount; i++){
        model->parameters[i] -= aux[i]; 
    }
} //gradient descent works, but this is way faster

void Train(Model *model, double **dataX, double *dataY,
           int setSize, int parameterAmount, long iterations, double learningRate, int polynomialOrder){
    model->setSize = setSize;
    model->parameterAmount = polynomialOrder * parameterAmount;
    model->parameters = malloc(sizeof(double) * model->parameterAmount);
    model->bias = 0;
    model->featuresMean = malloc(sizeof(double) * model->parameterAmount);
    model->featuresStandardDeviation = malloc(sizeof(double) * model->parameterAmount);
    AdamOptimizer *adam = CreateAdamOptimizer(model->parameterAmount);
    
    double **polynomialDataX = malloc(sizeof(double*) * setSize);
    for(int i = 0; i < setSize; i++){
        polynomialDataX[i] = malloc(sizeof(double) * model->parameterAmount);
        for(int j = 0; j < polynomialOrder; j++){
            for(int k = 0; k < parameterAmount; k++){
                polynomialDataX[i][j * parameterAmount + k] = pow(dataX[i][k], j + 1);
            }
        }
    }
    
    for(int i = 0; i < model->parameterAmount; i++){
        model->featuresMean[i] = ComputeFeatureMean(polynomialDataX, model, i);
        model->featuresStandardDeviation[i] = ComputeFeatureStandardDeviation(polynomialDataX, model, i);
    }
    
    //here you can choose to normalize the data
    //it's hardcoded because why would you not?
    
    //model->xTrain = polynomialDataX;
    ///*
    model->xTrain = ZScoreNormalize(polynomialDataX, model);
    
    for(int i = 0; i < setSize; i++)
        free(polynomialDataX[i]);
    free(polynomialDataX);
    //*/
    
    model->yTrain = dataY;
    
    for(int i = 0; i < iterations; i++)
        AdamOptimization(model, adam, learningRate, 0.9, 0.99, 0.000000001);
}

void PredictOutput(Model *model, double **input, int inputSize, 
                   double *output){
    for(int i = 0; i < inputSize; i++)
        output[i] = ComputeY(model, input[i]);
} //this is so simple, I love it :D
