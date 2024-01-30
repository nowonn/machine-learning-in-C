void FillDataArraysFromFile(FILE *file, double ***dataX, double **dataY, int *parameterAmount, int *setSize){
    fscanf(file, "%d%d", setSize, parameterAmount);
    
    *dataY = malloc(sizeof(double) * *setSize);
    *dataX = malloc(sizeof(double*) * *setSize);
    for(int i = 0; i < *setSize; i++){
        (*dataX)[i] = malloc(sizeof(double) * *parameterAmount);
    }
    
    for(int i = 0; i < *setSize; i++){
        for(int j = 0; j < *parameterAmount; j++)
            fscanf(file, "%lf,", &((*dataX)[i][j]));
        
        fscanf(file, "%lf", &((*dataY)[i]));
    }
    
    fclose(file);
}

double ComputeTrainingY(double *input, int parameterAmount, 
                        double *parameters, double bias, 
                        ModelType type){
    double result = bias;
    
    for(int i = 0; i < parameterAmount; i++)
        result += parameters[i] * input[i];
    
    return (type == POLYNOMIAL) ? result : Sigmoid(result);
}

void FillArrayWithRandomNumbers(double array[], int size, int x, int y) {
    for(int i = 0; i < size; i++) 
        array[i] = (rand() % (y - x + 1)) + x;
}

void FillArrayWithPointers(double **array, int size, int sizeOfArrays){
    for(int i = 0; i < size; i++){
        array[i] = malloc(sizeof(double) * sizeOfArrays);
        FillArrayWithRandomNumbers(array[i], sizeOfArrays, 0, 100);
    }
}

double EvaluatePolynomial(double *polynomial, int polynomialOrder, double x) {
    double result = 0;
    
    for (int i = polynomialOrder; i >= 0; i--)
        result = result * x + polynomial[i];
    
    return result;
}

void FillArrayWithFunction(double *array, int size, 
                           double *function, double functionB, 
                           double **input, int inputSize,
                           ModelType type,
                           double fluctuation){
    srand((unsigned int)time(NULL));
    for(int i = 0; i < size; i++){
        double random = (double)rand() / (double)RAND_MAX;
        array[i] = ComputeTrainingY(input[i], inputSize, function, functionB, type) + random * fluctuation;
    }
}

int GetIndexOfBiggest(double array[], int start, int end){
    int biggest = array[start];
    int index = start;
    for(int i = start + 1; i <= end; i++){
        if(array[i] > biggest){
            biggest = array[i];
            index = i;
        }
    }
    return index;
}
int GetIndexOfSmallest(double array[], int start, int end){
    int smallest = array[start];
    int index = start;
    for(int i = start + 1; i <= end; i++){
        if(array[i] < smallest){
            smallest = array[i];
            index = i;
        }
    }
    return index;
}

double GetBiggest(double array[], int start, int end){
    double biggest = array[start];
    for(int i = start + 1; i <= end; i++)
        if(array[i] > biggest) biggest = array[i];
    
    return biggest;
}

double GetSmallest(double array[], int start, int end){
    double smallest = array[start];
    for(int i = start + 1; i <= end; i++)
        if(array[i] < smallest) smallest = array[i];
    
    return smallest;
}

void FillArrayWithZeros(void *array, int size){
    memset(array, 0, size);
}