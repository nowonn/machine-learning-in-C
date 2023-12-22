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
    
    ModelType type = LOGISTIC;
    int parameterAmount = 4;
    
    int setSize = 100;
    double *dataX[setSize];
    double dataY[setSize];
    double functionBias = 0.02;
    double function[50] = {0.001, 0.0021, 0.0005, 0.012};
    
    bool done = false;
    long iterations = 0;
    long iterationsSinceDrawn = 12000000;
    //for it to draw on first iteration
    
    FillArrayWithPointers(dataX, setSize,
                          parameterAmount);
    
    Model model;
    model = CreateRegressionModel(type, dataX, dataY, setSize, parameterAmount);
    
    FillArrayWithFunction(dataY, setSize, function,
                          functionBias, model.xTrain,
                          parameterAmount, 
                          type, 0.001);
    
    AdamOptimizer *adam = CreateAdamOptimizer(model.parameterAmount);
    
    
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
        if(iterations < 2.5e5){
            if(iterationsSinceDrawn > 1000){
                XClearWindow(display, window);
                
                char buffer[25];
                FillArrayWithZeros(buffer, 25);
                sprintf(buffer, "iterations: %ldk", iterations/1000);
                XDrawString(display, window, gc, 70, 50, buffer, 25);
                
                DrawCost(display, window, gc, &model);
                XFlush(display);
                
                iterationsSinceDrawn = 0;
            }
            
            //GradientDescent(&model, 0.00001);
            AdamOptimization(&model, adam, 0.0001, 0.9, 0.99, 0.000000001);
            
            iterations++;
            iterationsSinceDrawn++;
        } else if(!done){
            
            XClearWindow(display, window);
            
            char buffer[25];
            FillArrayWithZeros(buffer, 25);
            sprintf(buffer, "iterations: %ld", iterations);
            XDrawString(display, window, gc, 70, 50, buffer, strlen(buffer));
            
            DrawCost(display, window, gc, &model);
            XFlush(display);
            done = true;
        }
    }
    
    XCloseDisplay(display);
    return 0;
}
