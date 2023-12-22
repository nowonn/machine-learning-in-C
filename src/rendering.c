Graph *CreateGraph(int startX, int startY, int endX, int endY,
                   int rulerXSpacing, int rulerYSpacing,
                   double **pointsX, double *pointsY,
                   int pointAmount){
    Graph *graph = malloc(sizeof(Graph));
    graph->startX = startX;
    graph->startY = startY;
    graph->endX = endX;
    graph->endY = endY;
    graph->rulerXSpacing = rulerXSpacing;
    graph->rulerYSpacing = rulerYSpacing;
    graph->pointsX = pointsX;
    graph->pointsY = pointsY;
    graph->pointAmount = pointAmount;
    
    return graph;
}

void DrawGraph(Display *display, Drawable window, GC gc, Graph *graph, Model *model, int chosenX){
    XDrawRectangle(display, window, gc,
                   graph->startX - 10,
                   graph->startY - 10, 
                   graph->endX - graph->startX + 10,
                   graph->endY - graph->startY - 10);
    for(int i = 0; i < graph->pointAmount; i++){
        if(windowHeight - graph->pointsY[i] < graph->startY << 1) continue;
        XDrawPoint(display, window, gc, graph->pointsX[i][chosenX] + graph->startX,
                   windowHeight - graph->pointsY[i] - graph->startY);
    }
}

void DrawModelCurve(Display *display, Drawable window, GC gc,
                    Graph *graph, Model *model){
    double *parameters = model->parameters;
    for(int x = 0; x < graph->endX - graph->startX; x++){
        double y1 = EvaluatePolynomial(parameters, model->parameterAmount - 1, x);
        double y2 = EvaluatePolynomial(parameters, model->parameterAmount - 1, x + 1);
        if(y1 < graph->startY || y2 > graph->endY - graph->startY) continue;
        XDrawLine(display, window, gc, 
                  graph->startX + x,
                  windowHeight - graph->startY - y1,
                  graph->startX + x + 1,
                  windowHeight - graph->startY - y2);
    }
}

void DrawCost(Display *display, Drawable window, GC gc, Model *model){
    char buffer[50];
    FillArrayWithZeros(buffer, 30);
    sprintf(buffer, "cost: %.5lf", ComputeCost(model));
    XDrawString(display, window, gc, 70, 70, buffer, strlen(buffer));
    
    FillArrayWithZeros(buffer, 30);
    sprintf(buffer, "bias: %.5lf", model->bias);
    XDrawString(display, window, gc, 70, 90, buffer, strlen(buffer));
    
    FillArrayWithZeros(buffer, 30);
    sprintf(buffer, "first 10 yTrains:", model->bias);
    XDrawString(display, window, gc, 380, 50, buffer, 17);
    
    FillArrayWithZeros(buffer, 30);
    sprintf(buffer, "first 10 guesses:", model->bias);
    XDrawString(display, window, gc, 500, 50, buffer, 17);
    
    FillArrayWithZeros(buffer, 30);
    sprintf(buffer, "guessed parameters:", model->bias);
    XDrawString(display, window, gc, 70, 110, buffer, 19);
    
    for(int i = 0; i < model->parameterAmount; i++){
        FillArrayWithZeros(buffer, 30);
        sprintf(buffer, "%.5lf", model->parameters[i]);
        XDrawString(display, window, gc, 70, 130 + 20*i, buffer, strlen(buffer));
    }
    
    for(int i = 0; i < 10; i++){
        FillArrayWithZeros(buffer, 30);
        sprintf(buffer, "%.5lf", model->yTrain[i]);
        XDrawString(display, window, gc, 380, 70 + 20*i, buffer, strlen(buffer));
    }
    
    for(int i = 0; i < 10; i++){
        FillArrayWithZeros(buffer, 30);
        sprintf(buffer, "%.5lf", ComputeY(model, model->xTrain[i]));
        XDrawString(display, window, gc, 500, 70 + 20*i, buffer, strlen(buffer));
    }
}