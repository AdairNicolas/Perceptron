#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_INPUTS 2 //Definimos el numero de entradas
#define LEARNING_RATE 0.1 //Definimos learning rate
#define NUM_EPOCHS 1000 //Definimos limite de epocas

float weights[NUM_INPUTS]; //Arreglo que contendrá a los pesos
float bias;
float rest;
int continuar = 1;
float cost = 0;
int training_data[][NUM_INPUTS + 1] = { //Datos de entrenamiento
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1}
};

int desire_outputs[4] = {0,0,0,1};
float predictions[4] = {0,0,0,0};

float get_v(float inputs[], float weights[], float bias) { //suma de bias  mas producto de pesos con entradas
    float result = bias;
    for (int i = 0; i < NUM_INPUTS; i++) {
        result += inputs[i] * weights[i];
    }
    return result;
}

int umbral(float input) { //Umbral
    return input >=  0.500000000? 1 : 0;
}

float sigmoid(float x) { //funcion de activacion sigmoide
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivada(double x) { //Derivada de funcion de activación sigmoide
    return sigmoid(x) * (1.0 - sigmoid(x));
}

void train_with_delta() {
	continuar = 1;
    int total_errors = 1;  //Inicializamos variable de total de errores
    for (int epoch = 1; epoch < NUM_EPOCHS && continuar != 0; epoch++) { //For loop que estará activa mientras no se llegue al limite de épocas (100) o que se complete una epoca con funciónde perdida menor a 0.5
        printf("\nEpoch: %d\n",  epoch);
        for (int i = 0; i < 4; i++) { // Inicio de epoca
            float inputs[NUM_INPUTS];
            for (int j = 0; j < NUM_INPUTS; j++) {
                inputs[j] = training_data[i][j];    //Leemos las inputs de los datos de entrenamiento
            }
            float v = weights[0]*inputs[0] + weights[1]*inputs[1] + bias;
            int target = training_data[i][NUM_INPUTS];   //Obtenermos el target de datos de entrenamiento
            float prediction = sigmoid(v); //Función de activación con el resultado del producto punto
            predictions[i] = prediction;
            float error = target - prediction; //Obtencion de error
            bias += LEARNING_RATE * error *sigmoid_derivada(v); //Modificación de bias
            for (int j = 0; j < NUM_INPUTS; j++) {
                weights[j] += LEARNING_RATE * error * inputs[j] * sigmoid_derivada(v); //Modificación de pesos
            }	
            
            printf("Weights: %f %f, Inputs: %f %f\n", 
            weights[0], weights[1], inputs[0], inputs[1]);
            printf("Target: %d, Prediction: %f, Error: %f, Bias:%f\n", target, prediction, error, bias);
            
        }  //fin de epoca
        cost = 0;
        for (int i = 0; i < 4; ++i)//Función de costo
        {
        	rest = desire_outputs[i] - predictions[i];
        	cost += pow(rest, 2);
        }
        if (cost < 0.5)
        {
        	continuar = 0;
        }
    }
}

int main() {

    float inputs[][NUM_INPUTS] = { //Datos para el test
      {0, 0},
      {0, 1},
      {1, 0},
      {1, 1}
    };

    srand(time(NULL));
    for (int i = 0; i < NUM_INPUTS; i++) {
        weights[i] = ((float)rand()/(float)(RAND_MAX)) * 2 - 1; // Inicializar pesos random en 1 y -1
    }
    bias = ((float)rand()/(float)(RAND_MAX)) * 2 - 1; // Inicializar bias random entre 1 y -1
    
    printf("\nOR pesos y bias aleatorio\n");
    printf("W1: %f, W2: %f, Bias: %f\n", weights[0], weights[1], bias);
    
    train_with_delta(); //Entrenamiento de perceptron
    
    printf("\nWeights: %f, %f, bias: %f\n", weights[0], weights[1], bias);
    for(int m = 0; m < 4;m++){
    	float v = get_v(inputs[m], weights, bias); 
    	float active = sigmoid(v); //Predicción final
        int prediction = umbral(active);
        printf("Prediction for [%f, %f]: %d\n", inputs[m][0], inputs[m][1], prediction);
    }

}

