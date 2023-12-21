#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>

bool running = true;
int windowWidth = 800;
int windowHeight = 600;

#include "nowonmllib.h"
#include "utils.c"
#include "rendering.c"

int main() {
    Display *display;
    Window window;
    XEvent event;
    int screen;
    XGCValues values;
    GC gc;
    unsigned long valuemask = 0;
    
    display = XOpenDisplay(NULL);
    if (display == NULL) {
        fprintf(stderr, "Cannot open display\n");
        return 1;
    }
    
    screen = DefaultScreen(display);
    int depth = DefaultDepth(display, screen);
    Visual *visual = DefaultVisual(display, screen);
    XSetWindowAttributes attributes;
    window = XCreateWindow(display, RootWindow(display, screen), 10, 10, windowWidth, windowHeight, 1,
                           depth, InputOutput, visual, valuemask, &attributes);
    XSelectInput(display, window, ExposureMask | KeyPressMask | StructureNotifyMask);
    XMapWindow(display, window);
    
    gc = XCreateGC(display, window, valuemask, &values);
    if (gc < 0) {
        fprintf(stderr, "XCreateGC: \n");
    }
    
    XSetLineAttributes(display, gc, 1, LineSolid, CapButt, JoinMiter);
    XSetForeground(display, gc, WhitePixel(display, screen));
    XSetWindowBackground(display, window, BlackPixel(display, screen));
    /////////////////////////////////////////////////////////
    srand(time(NULL));
    
    int setSize = 100;
    double *dataX[100];
    double dataY[100];
    int parameterAmount = 4;
    double parameters[50];
    double bias;
    double functionBias = 0.24;
    double function[50] = {0.001, 0.002, 0.005, 0.013};
    long iterations = 0;
    long iterationsSinceDrawn = 12000000;
    //for it to draw on first iteration
    
    Model *model;
    //model = CreateLinearRegressionModel(dataX, dataY, setSize, parameters, parameterAmount, bias);
    model = CreateLogisticRegressionModel(dataX, dataY, setSize, parameters, parameterAmount, bias);
    
    AdamOptimizer *adam = CreateAdamOptimizer(model->parameterAmount);
    
    FillArrayWithPointers(dataX, model->setSize,
                          model->parameterAmount);
    FillArrayWithFunction(dataY, model->setSize, function,
                          functionBias, dataX,
                          model->parameterAmount, 
                          model->type, 0.01);
    
    while (running) {
        while (XPending(display) > 0) {
            XNextEvent(display, &event);
            if(event.type == KeyPress) {
                KeySym key = XLookupKeysym(&event.xkey, 0);
                if(key == XK_Escape) {
                    running = false;
                }
            } else if(event.type == ConfigureNotify) {
                XConfigureEvent xce = event.xconfigure;
                
                windowWidth = xce.width;
                windowHeight = xce.height;
            }
        }
        
        if(iterationsSinceDrawn > 10000){
            XClearWindow(display, window);
            char buffer[25];
            FillArrayWithZeros(buffer, 25);
            sprintf(buffer, "iterations: %ld million", iterations/1000000);
            XDrawString(display, window, gc, 70, 50, buffer, 25);
            DrawCost(display, window, gc, model);
            XFlush(display);
            iterationsSinceDrawn = 0;
        }
        
        //GradientDescent(model, 0.00001);
        AdamOptimization(ComputeCostLinear, model, adam, 0.000001, 0.9, 0.99, 0.000000001);
        iterations++;
        iterationsSinceDrawn++;
    }
    
    XCloseDisplay(display);
    return 0;
}
