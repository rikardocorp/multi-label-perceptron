//
//  mlp.cpp
//  mlp
//
//  Created by Ricardo Coronado on 18/04/16.
//  Copyright © 2016 Ricardo Coronado. All rights reserved.
//

#include <iostream>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <ctime>

using namespace std;

//  ESTRUCTURA PARA DATA DE ENTRENAMIENTO Y TEST

struct _datos
{
    float * data = NULL;
    int target;
    struct _datos *sig = NULL;
};

//  FUNCIONES

void obtener_data(string,int);
void liberar_memoria();
void imprimir_array();
void mlp();
void mlp_test();
void copiar_input(_datos *origen, int cantidad);
float f_signoid(float numero);
void multiplicacion_punto(int ,int ,int , int );
void generar_deltas(int pos_target,int pos, int capaAux);
void actualizar_pesos(int posInput, int posDelta, int cant_input, int cant_deltas, int numeroCapa);

//  VARIABLES GLOBALES

_datos *point_data_fit,*point_data_test;
float *point_capa;
float *capa;
int *neurona;
int total_capas;
int total_input;
int total_salidas;
int *pos_ini;
int total_array;
float learn_rate;
float **sumatorias_error;
float target[8][8] = {{0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01},
                    {0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01},
                    {0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01},
                    {0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01},
                    {0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01},
                    {0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01},
                    {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01},
                    {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99}};
//float target[2][2] ={{0.99,0.01},{0.01,0.99}};
//float target[3][3] ={{0.99,0.01,0.01},{0.01,0.99,0.01},{0.01,0.01,0.99}};


//  VARIABLES PARA EL CLOCK

clock_t start;
double duration;

int main(int argc, const char * argv[])
{

//  INICIALIZAMOS VARIABLES
//  _______________________
    
    total_capas = 3;
    neurona = (int *)malloc(sizeof(int)*total_capas+1);
    
    //  Estructura de la neurona el indice del array neurona es el numero de capas y el valor es la cantidad de neuronas
    neurona[0] = 58;
    neurona[1] = 9;
    neurona[2] = 8;
    neurona[3] = 0;
    total_input = neurona[0];
    total_salidas = neurona[total_capas-1];
    learn_rate = 0.9;
    
    //  Array para almacenar posiciones de inicio de las capas
    pos_ini = (int *)malloc(sizeof(int)*total_capas);

    //  inicializamos las posiciones de las capas - y hallamos el total de elementos del array
    int total = 0;
    for (int i = 0; i < total_capas; i++)
    {
        pos_ini[i] = total;
        total = total + neurona[i] + 1 + (neurona[i]+1) * (neurona[i+1]);
    }
    
    //  inicializamos el array principal de elementos con valores random
    total_array = total;
    srand(time(0));
    capa = (float *)malloc(sizeof(float)*total_array);
    for (int i = 0; i< total; i++) {
        capa[i] = (double) rand()/RAND_MAX;
    }
    
    //  Reservamos espacio en memoria para una matriz que almacene el error acumulado de una neurona
    sumatorias_error = (float **)malloc(sizeof(float)*(total_capas-1));
    for (int i = 0; i< total_capas-1; i++) {
        sumatorias_error[i] = (float *)malloc(sizeof(float)*(neurona[i]+1));
        for (int j=0; j < neurona[i]+1; j++) {
            sumatorias_error[i][j] = 0;
        }
    }
    
    //  Puntero al array principal
    point_capa = capa;

    
//  ENTRENAMIENTO
//  _____________
    
    //  Generamos la data en una estructura de listas enlazadas
    obtener_data("cara.csv",total_input);
    //  Algoritmo MLP
    mlp();
    
    
//  TESTEO
//  ______
    
    //  Generamos la data en una estructura de listas enlazadas
    obtener_data("cara_test.csv",total_input);
    //  Corremos el Forward Propagation
    mlp_test();
    
    
//  LIBERAMOS MEMORIA
//  _________________
    
    
    liberar_memoria();
    point_data_fit = NULL;
    free(point_data_fit);
    point_capa = NULL;
    free(point_capa);
    return 0;
}


//  MLP FORWARD Y BACKWARD PROPAGATION
void mlp()
{
    int total_epocas = 3000;
    _datos *data;
    
    start = clock();
    
    //  Recorremos las epocas
    for (int epoca = 0; epoca< total_epocas; epoca++)
    {
        //  Recorremos la data que esta almacenada en una lista enlazada
        data = point_data_fit;
        while (data != NULL)
        {
            //  Inicializamos el array con los datos de entrada del caso de entrenamiento
            copiar_input(data,total_input);
            
            //  FORWARD PROPAGATION
            for (int c = 0; c < total_capas-1; c++)
            {
                multiplicacion_punto(pos_ini[c], pos_ini[c+1], neurona[c], neurona[c+1]);
            }
            
            //  BACKWARD PROPAGATION
            for (int c = total_capas-2; c > -1; c--)
            {
                if (c == total_capas-2)
                {
                    generar_deltas(data->target-1, pos_ini[c+1], neurona[c+1]);
                }
                actualizar_pesos(pos_ini[c],pos_ini[c+1],neurona[c],neurona[c+1],c);
            }
            
            data = data->sig;
        }
    }
    
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
}


//  MLP TEST FORWARD PROPAGATION
void mlp_test()
{
    _datos *data = point_data_fit;
    int res;
    int pos_resultado = pos_ini[total_capas - 1] + 1;
    string salida;
    
    int contador = 0;
    int salidaFinal = 0;
    int contadorCasos = 0;
    float acurracy = 0;
    
    while (data != NULL) {
        point_capa = capa;
        copiar_input(data,total_input);
        res = data->target;
        
        //  Forward Propagation
        for (int c = 0; c < total_capas-1; c++)
        {
            multiplicacion_punto(pos_ini[c], pos_ini[c+1], neurona[c], neurona[c+1]);
        }
        
        contadorCasos++;
        for (int i = 0; i < total_salidas; i++)
        {
            
            salidaFinal = (int)(point_capa[pos_resultado+i]+0.5);
            if(res-1 == i && salidaFinal == 1)
            {
                contador++;
            }
            cout<<salidaFinal<<" - ";
        }
        
        cout<<"- "<<res<<endl;
        data = data->sig;
    }
    
    acurracy = contador * 100 / contadorCasos;
    cout<<endl<<endl<<"Tiempo de Entrenamiento: "<< duration <<endl;
    cout<<"Total de Casos: "<<contadorCasos<<endl;
    cout<<"Aciertos: "<<contador<<endl;
    cout<<"Desaciertos: "<<contadorCasos - contador<<endl;
    cout<<"Accuracy: "<<acurracy<<"%"<<endl;
    
}


//  INICIALIZAMOS LAS ENTRADAS PARA LA PRIMERA CAPA DE CADA CASO DE ENTRENAMIENTO
void copiar_input(_datos *origen, int cantidad)
{
    point_capa[0] = 1;
    for (int i=1; i<cantidad+1; i++) {
        point_capa[i] = origen->data[i-1];
    }
}


//  GENERAMOS LOS DELTAS PARA LA PRIMERA ITERACION DEL BACKWARD PROPAGATION
void generar_deltas(int pos_target,int pos, int capaAux)
{
    for (int i = 1; i < capaAux+1; i++) {
        float out = point_capa[pos+i];
        float obj = target[pos_target][i-1];
        point_capa[pos+i] = ( out - obj) * out * (1 - out);
    }
}

//  ACTUALIZAR PESOS
void actualizar_pesos(int posInput, int posDelta, int cant_input, int cant_deltas,int numeroCapa){
    // indicie para marcar el inicio de la seccion de pesos de la capa i
    int ini_matriz = posInput + cant_input + 1;
    float derivada;
    
    //  Recorremos los deltas de la capa que hallamos anteriormente - hay un delta por neurona
    for (int i=1; i < cant_deltas+1; i++)
    {
        //  Recorremos las entradas de las neuronas de la capa siguiente
        for (int j=0; j < cant_input+1; j++)
        {
            //  Acumulamos el error cometido para casa neurona j esima de la capa l
            sumatorias_error[numeroCapa][j] = sumatorias_error[numeroCapa][j] + (point_capa[posDelta+i] * point_capa[ini_matriz+j]);
            //  La derivada del error total respecto a un peso wi es igual al delta x por la entrada de la neurona relacionada al peso
            derivada = point_capa[posDelta + i] * point_capa[posInput+j];
            //  Actualizamos el peso
            point_capa[ini_matriz + j] = point_capa[ini_matriz + j] - learn_rate * derivada;
        }
        
        //  como utilizamos un array para almacenar toda la data debemos actualizar el indice para simular el comportamiento de matriz
        ini_matriz = ini_matriz + cant_input + 1;
    }
    
    //  Por generalizacion acumulamos el error del bias que se almacena en la posicion 0, pero esto no es necesario ya que el bias es una neurona aislada, la primera posicion la inicializamos a 0 para las proximas iteraciones
    sumatorias_error[numeroCapa][0] = 0;
    for (int i=1; i < cant_input + 1; i++)
    {
        //  Almacenamos el delta para la iteracion de la siguiente capa, en nuestro array esto lo almacenamos en los input de salida de la proxima capa ya que estos no se volveran a utilizar, de esta manera reducimos el tamaño de nuestro array
        point_capa[posInput + i] = sumatorias_error[numeroCapa][i] * (point_capa[posInput + i] * (1 - point_capa[posInput + i]));
        //  Inicializamos el acumulador a 0 para proximas iteraciones
        sumatorias_error[numeroCapa][i] = 0;
    }
}


//  SIMULACION DE MULTIPLICACION DE MATRICES
    //  pos_input -> manda la posicion inicial de las input para la capa
    //  pos_peso -> manda la posicion inicial de los pesos
    //  capa_iniput -> cantidad de Inputs que alimentan a la capa actual = a la cantidad de neuronas de la capa anterior
    //  capa_peso -> cantidad de pesos que recibe la neurona de la capa actual
void multiplicacion_punto(int pos_input,int pos_peso,int capa_input, int capa_peso){
    float sumatoria;
    int ini_matriz = capa_input + 1;
    point_capa[pos_peso] = 1;

    for (int i=1; i<capa_peso+1; i++){
        sumatoria = 0;
        for (int j=0; j<capa_input+1; j++){
            sumatoria = sumatoria + point_capa[pos_input+j]* point_capa[pos_input+ini_matriz*i+j];
        }
        point_capa[pos_peso+i] = f_signoid(sumatoria);
    }
}


//  FUNCION DE ACTIVACION
float f_signoid(float numero)
{
    return 1 / (1 + pow(exp(1), -1 * numero));
}


//  GENERAMOS UNA ESTRUCTURA CON LA DATA DE ENTRENAMIENTO O TEST
void obtener_data(string archivo,int total_input){
    
    string numero;
    int a,b;
    _datos *nuevo, *puntero_aux;
    
    nuevo = (struct _datos *) malloc (sizeof(struct _datos));
    puntero_aux = nuevo;
    point_data_fit = nuevo;
    
    
    ifstream file(archivo);
    string value;
    while (file.good())
    {
        getline ( file, value, '\n' );
        
        //  declaramos un array
        float *d_array;
        d_array = (float *)malloc(sizeof(float)*total_input);
        int c = 0;
        
        //  recorremos cada linea del file
        for (int i=0; i<total_input; i++)
        {
            a = (int)value.find(';',c);
            b = (int)value.find(';',a+1);
            c = b;
            numero = value.substr(a+1,b-a-1);
            d_array[i] = stof(numero);
        }
        
        //  almacenamos los datos en la lista y enlazamos
        a = (int)value.find(';');
        numero = value.substr(0,a);
        nuevo->target = stoi(numero);
        nuevo->data = d_array;
        puntero_aux->sig = nuevo;
        nuevo->sig = NULL;
        puntero_aux = nuevo;
        
        nuevo = (struct _datos *) malloc (sizeof(struct _datos));
    }
    
    //  liberamos la memoria
    puntero_aux = NULL;
    free(nuevo);
    free(puntero_aux);
}

void liberar_memoria()
{
    _datos *puntero_aux,*puntero_aux2;
    puntero_aux = point_data_fit;
    puntero_aux2 = point_data_fit;
    while (puntero_aux2 != NULL)
    {
        puntero_aux = puntero_aux2;
        puntero_aux2 = puntero_aux2->sig;
        free(puntero_aux->data);
        free(puntero_aux);
    }

    puntero_aux = NULL;
    puntero_aux2 = NULL;
    free(puntero_aux);
    free(puntero_aux2);
}

void imprimir_array()
{
    for (int i = 0; i < total_array; i++) {
        cout<<point_capa[i]<<"|";
    }
}





