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
    
    double *dataX[100];
    double dataY[100];
    int paramaterAmount = 5;
    double parameters[50] = {0.5, 4, 1, 1, 1};
    double bias = 30;
    double function[50] = {2, 0.5, 0.1, 4, 3};
    double functionBias = 50;
    long iterations = 0;
    long iterationsSinceDrawn = 12000000;
    
    Model *model = 
        CreateLinearRegressionModel(dataX, dataY, 100, parameters, paramaterAmount, bias);
    Graph *graph = CreateGraph(windowWidth / 20, windowHeight / 20,
                               windowWidth - (windowWidth / 20),
                               windowHeight - (windowHeight / 20),
                               10, 10,
                               dataX, dataY, 100);
    AdamOptimizer *adam = CreateAdamOptimizer(model->parameterAmount);
    
    FillArrayWithPointers(dataX, model->setSize, model->parameterAmount);
    FillArrayWithFunction(dataY, model->setSize, function, functionBias, dataX, model->parameterAmount, 10);
    
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
                
                XClearWindow(display, window);
                XFlush(display);
                
                graph->startX = windowWidth / 20;
                graph->startY = windowHeight / 20;
                graph->endX = windowWidth - (windowWidth / 20);
                graph->endY = windowHeight - (windowHeight / 20);
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
        AdamOptimization(ComputeCostLinear, model, adam, 0.0001, 0.9, 0.99, 0.000000001);
        iterations++;
        iterationsSinceDrawn++;
    }
    
    XCloseDisplay(display);
    return 0;
}
