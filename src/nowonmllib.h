/* date = December 16th 2023 10:50 am */

#ifndef NOWONMLLIB_H
#define NOWONMLLIB_H

typedef enum { LINEAR, LOGISTIC } ModelType;

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
} Model;

typedef struct {
    double *m;
    double *v;
    double mBias;
    double vBias;
    int t;
} AdamOptimizer;

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

Model CreateRegressionModel(ModelType type,
                            double **xTrain, double *yTrain,
                            int setSize, int parameterAmount){
    Model model;
    model.type = type;
    model.setSize = setSize;
    model.parameterAmount = parameterAmount;
    model.parameters = malloc(sizeof(double) * parameterAmount);
    model.bias = 0;
    model.featuresMean = malloc(sizeof(double) * parameterAmount);
    model.featuresStandardDeviation = malloc(sizeof(double) * parameterAmount);
    
    for(int i = 0; i < parameterAmount; i++){
        model.featuresMean[i] = ComputeFeatureMean(xTrain, &model, i);
        model.featuresStandardDeviation[i] = ComputeFeatureStandardDeviation(xTrain, &model, i);
    }
    
    //model.xTrain = xTrain;
    model.xTrain = ZScoreNormalize(xTrain, &model);
    model.yTrain = yTrain;
    
    return model;
} 

double Sigmoid(double x){
    return 1/(1 + exp(-x));
}

double ComputeY(Model *model, double *input){
    double result = model->bias;
    
    for(int i = 0; i < model->parameterAmount; i++)
        result += model->parameters[i] * input[i];
    
    return (model->type == LINEAR) ? result : Sigmoid(result);
}

double MeanSquaredErrorLoss(Model *model, double guess, double expectedOutput){
    return ((guess - expectedOutput)/2)*(guess - expectedOutput);
}

double LogisticLoss(Model *model, double guess, double expectedOutput){
    return -expectedOutput * log(guess) - (1 - expectedOutput) * log(1 - guess);
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
    
    return totalCost/setSize;
}

double PartiallyDeriveCost(Model *model, int parameterIndex, bool isBias){
    double h = 0.0000000001;
    double *parameter;
    if(!isBias) parameter = &model->parameters[parameterIndex];
    else parameter = &model->bias;
    
    double temp = *parameter;
    
    *parameter = temp + h;
    double cost1 = ComputeCost(model);
    
    *parameter = temp - h;
    double cost2 = ComputeCost(model);
    
    *parameter = temp;
    
    return (cost1 - cost2)/(2 * h);
}

void GradientDescent(Model *model, double learningRate){
    double aux[model->parameterAmount];
    
    for(int i = 0; i < model->parameterAmount; i++){
        double derivative = PartiallyDeriveCost(model, i, 0);;
        aux[i] = derivative * learningRate; 
    }
    
    double derivative = PartiallyDeriveCost(model, 0, 1);
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
        double g = PartiallyDeriveCost(model, i, 0);
        optimizer->m[i] = beta1 * optimizer->m[i] + (1 - beta1) * g;
        optimizer->v[i] = beta2 * optimizer->v[i] + (1 - beta2) * g * g;
        double mHat = optimizer->m[i] / (1 - pow(beta1, optimizer->t));
        double vHat = optimizer->v[i] / (1 - pow(beta2, optimizer->t));
        aux[i] = learningRate * mHat / (sqrt(vHat) + epsilon);
    }
    
    double gBias = PartiallyDeriveCost(model, 0, 1);
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
#endif //NOWONMLLIB_H
