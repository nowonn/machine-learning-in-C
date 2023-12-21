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
} Model;

typedef struct {
    double *m;
    double *v;
    double mBias;
    double vBias;
    int t;
} AdamOptimizer;

Model *CreateLinearRegressionModel(double **xTrain,
                                   double *yTrain,
                                   int setSize,
                                   double *parameters,
                                   int parameterAmount,
                                   double bias){
    Model *model = malloc(sizeof(Model));
    model->type = LINEAR;
    model->xTrain = xTrain;
    model->yTrain = yTrain;
    model->setSize = setSize;
    model->parameters = parameters;
    model->parameterAmount = parameterAmount;
    model->bias = bias;
    
    return model;
} 

Model *CreateLogisticRegressionModel(double **xTrain,
                                     double *yTrain,
                                     int setSize,
                                     double *parameters,
                                     int parameterAmount,
                                     double bias){
    Model *model = malloc(sizeof(Model));
    model->type = LOGISTIC;
    model->xTrain = xTrain;
    model->yTrain = yTrain;
    model->setSize = setSize;
    model->parameters = parameters;
    model->parameterAmount = parameterAmount;
    model->bias = bias;
    
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

double ComputeCostLinear(Model *model){
    double **input = model->xTrain;
    double *expectedOutput = model->yTrain;
    int size = model->setSize;
    double *parameters = model->parameters;
    int parameterAmount = model->parameterAmount;
    double bias = model->bias;
    double totalCost = 0;
    for(int i = 0; i < size; i++){
        double guess = ComputeY(model, input[i]);
        totalCost += (guess - expectedOutput[i])*(guess - expectedOutput[i]);
    }
    
    return totalCost/(2*size);
}//low precision because of floating point shenanigans

double PartiallyDeriveCost(double (ComputeCost)(Model*), Model *model, int parameterIndex, bool isBias){
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

void GradientDescent(double (ComputeCost)(Model*), Model *model, double learningRate){
    double aux[model->parameterAmount];
    
    for(int i = 0; i < model->parameterAmount; i++){
        double derivative = PartiallyDeriveCost(ComputeCost, model, i, 0);;
        aux[i] = derivative * learningRate; 
    }
    
    double derivative = PartiallyDeriveCost(ComputeCost, model, 0, 1);
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
void AdamOptimization(double (ComputeCost)(Model*), Model *model, AdamOptimizer *optimizer, double learningRate, double beta1, double beta2, double epsilon){
    double aux[model->parameterAmount];
    optimizer->t += 1;
    for(int i = 0; i < model->parameterAmount; i++){
        double g = PartiallyDeriveCost(ComputeCost, model, i, 0);
        optimizer->m[i] = beta1 * optimizer->m[i] + (1 - beta1) * g;
        optimizer->v[i] = beta2 * optimizer->v[i] + (1 - beta2) * g * g;
        double mHat = optimizer->m[i] / (1 - pow(beta1, optimizer->t));
        double vHat = optimizer->v[i] / (1 - pow(beta2, optimizer->t));
        aux[i] = learningRate * mHat / (sqrt(vHat) + epsilon);
    }
    
    double g_bias = PartiallyDeriveCost(ComputeCost, model, 0, 1);
    optimizer->mBias = beta1 * optimizer->mBias + (1 - beta1) * g_bias;
    optimizer->vBias = beta2 * optimizer->vBias + (1 - beta2) * g_bias * g_bias;
    double mHat_bias = optimizer->mBias / (1 - pow(beta1, optimizer->t));
    double vHat_bias = optimizer->vBias / (1 - pow(beta2, optimizer->t));
    double aux_bias = learningRate * mHat_bias / (sqrt(vHat_bias) + epsilon);
    
    model->bias -= aux_bias;
    
    for(int i = 0; i < model->parameterAmount; i++){
        model->parameters[i] -= aux[i]; 
    }
} //gradient descent works, but this is way faster
#endif //NOWONMLLIB_H
