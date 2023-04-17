#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_INPUTS 2 //Definimos el numero de entradas
#define LEARNING_RATE 0.1 //Definimos learning rate
#define NUM_EPOCHS 100 //Definimos limite de epocas

float weights[NUM_INPUTS]; //Arreglo que contendrá a los pesos
float bias;
int training_data[][NUM_INPUTS + 1] = { //Datos de entrenamiento
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0}
};

float dot_product(float inputs[], float weights[], float bias) { //Producto punto de entradas por pesos
    float result = bias;
    for (int i = 0; i < NUM_INPUTS; i++) {
        result += inputs[i] * weights[i];
    }
    return result;
}

int activation(float input) { //Función de activación
    return input >= -0.000001 ? 1 : 0;
}

void train_with_delta() {
    int total_errors = 1;  //Inicializamos variable de total de errores
    for (int epoch = 1; epoch < NUM_EPOCHS && total_errors != 0; epoch++) { //For loop que estará activa mientras no se llegue al limite de épocas (100) o que se complete una epoca sin errores
        printf("\nEpoch: %d\n",  epoch);
        total_errors = 0;  // reset total_errors para cada ciclo
        for (int i = 0; i < 4; i++) {
            float inputs[NUM_INPUTS];
            for (int j = 0; j < NUM_INPUTS; j++) {
                inputs[j] = training_data[i][j];    //Leemos las inputs de los datos de entrenamiento
            }
            int target = training_data[i][NUM_INPUTS];   //Obtenermos el target de datos de entrenamiento
            float result = dot_product(inputs, weights, bias);  //Obtenemos producto punto
            float prediction = activation(result); //Función de activación con el resultado del producto punto
            float error = target - prediction; //Obtencion de error
            bias += LEARNING_RATE * error; //Modificación de bias
            for (int j = 0; j < NUM_INPUTS; j++) {
                weights[j] += LEARNING_RATE * error * inputs[j]; //Modificación de pesos
            }
            
            printf("Weights: %f %f, Inputs: %f %f, Result: %f\n", 
            weights[0], weights[1], inputs[0], inputs[1], result);
            printf("Target: %d, Prediction: %f, Error: %f, Bias:%f\n", target, prediction, error, bias);
            
            // verificar si errror es igual a 0, caso contrario sumar en uno al total de errores
            if (error != 0) {
                total_errors++;
            }
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
    
    printf("\nXOR pesos y bias aleatorio\n");
    printf("W1: %f, W2: %f, Bias: %f\n", weights[0], weights[1], bias);
    
    train_with_delta(); //Entrenamiento de perceptron
    
    printf("\nWeights: %f, %f, bias: %f\n", weights[0], weights[1], bias);
    for(int m = 0; m < 4;m++){
        int prediction = activation(dot_product(inputs[m], weights, bias));
        printf("Prediction for [%f, %f]: %d\n", inputs[m][0], inputs[m][1], prediction);
    }
    weights[0] = 1;
    weights[1] = 0;
    bias = 0.5;
    
    printf("\nXOR pesos y bias definidos\n");
    printf("W1: %f, W2: %f, Bias: %f\n", weights[0], weights[1], bias);

    train_with_delta(); //Entrenamiento de perceptron

    printf("\nWeights: %f, %f, bias: %f\n", weights[0], weights[1], bias);
    for(int m = 0; m < 4;m++){
        int prediction = activation(dot_product(inputs[m], weights, bias));
        printf("Prediction for [%f, %f]: %d\n", inputs[m][0], inputs[m][1], prediction);
}

    return 0;
}

