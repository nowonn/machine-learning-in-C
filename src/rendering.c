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