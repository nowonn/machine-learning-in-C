#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>

bool paused = false;
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
    XSync(display, False);
    
    gc = XCreateGC(display, window, valuemask, &values);
    if (gc < 0) {
        fprintf(stderr, "XCreateGC: \n");
    }
    
    XSetLineAttributes(display, gc, 1, LineSolid, CapButt, JoinMiter);
    XSetForeground(display, gc, WhitePixel(display, screen));
    XSetWindowBackground(display, window, BlackPixel(display, screen));
    
    XClearWindow(display, window);
    
    char buffer[25];
    FillArrayWithZeros(buffer, 25);
    sprintf(buffer, "training...");
    XDrawString(display, window, gc, 70, 50, buffer, strlen(buffer));
    
    XFlush(display);
    
    /////////////////////////////////////////////////////////
    srand(time(NULL));
    
    ModelType type = POLYNOMIAL;
    double regularizationParameter = 0;
    long iterations = 500000;
    double learningRate = 0.0001;
    int polynomialOrder = 1;
    
    int parameterAmount;
    int setSize;
    double **dataX;
    double *dataY;
    FILE *inputFile = fopen("../data/input.csv", "r");
    
    FillDataArraysFromFile(inputFile, &dataX, &dataY, &parameterAmount, &setSize);
    
    Model *model;
    model = CreateRegressionModel(type,
                                  regularizationParameter);
    
    Train(model, dataX, dataY, setSize, parameterAmount, iterations, learningRate, polynomialOrder);
    
    XEvent e;
    do {
        XNextEvent(display, &e);
    } while (e.type != Expose);
    
    XClearWindow(display, window);
    
    FillArrayWithZeros(buffer, 25);
    sprintf(buffer, "iterations: %ld", iterations);
    XDrawString(display, window, gc, 70, 50, buffer, strlen(buffer));
    
    DrawCost(display, window, gc, model);
    XFlush(display);
    
    while (running) {
        while (XPending(display) > 0) {
            XNextEvent(display, &event);
            if(event.type == KeyPress) {
                KeySym key = XLookupKeysym(&event.xkey, 0);
                if(key == XK_Escape) running = false;
                if(key == XK_space){
                    Train(model, dataX, dataY, setSize, parameterAmount, iterations, learningRate, polynomialOrder);
                    
                    XClearWindow(display, window);
                    
                    FillArrayWithZeros(buffer, 25);
                    sprintf(buffer, "iterations: %ld", iterations);
                    XDrawString(display, window, gc, 70, 50, buffer, strlen(buffer));
                    
                    DrawCost(display, window, gc, model);
                    XFlush(display);
                    
                }
            } else if(event.type == ConfigureNotify) {
                XConfigureEvent xce = event.xconfigure;
                
                windowWidth = xce.width;
                windowHeight = xce.height;
            }
        }
        
    }
    
    XCloseDisplay(display);
    return 0;
}
