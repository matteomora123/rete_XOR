#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUTS 3
#define NEURONS_1 5
#define OUTPUTS 1

#define LEARNING_RATE 0.001 // Learning rate for weight updates
#define EPOCHS 5000 // Number of epochs for training
#define ERROR_THRESHOLD 0.001 // Error threshold for stopping criteria

#define FULL_DATASET_SIZE 8
#define TRAIN_DATASET_SIZE 5
#define TEST_DATASET_SIZE (FULL_DATASET_SIZE - TRAIN_DATASET_SIZE)

typedef struct {
    double inputs[INPUTS];

    double weights_1[INPUTS*NEURONS_1];
    double d_weights_1[INPUTS*NEURONS_1];

    double bias_1[NEURONS_1];
    double d_bias_1[NEURONS_1];

    double input_neurons_1[NEURONS_1];
    double output_neurons_1[NEURONS_1];
    double error_neurons_1[NEURONS_1];

    double weights_2[NEURONS_1*OUTPUTS];
    double d_weights_2[NEURONS_1*OUTPUTS];
    
    double bias_2[OUTPUTS];
    double d_bias_2[OUTPUTS];

    double input_outputs[OUTPUTS];
    double output_outputs[OUTPUTS];
    double error_outputs[OUTPUTS];

} NeuralNewtork;

void mat_mult(const double *A, const double *B, double *C, const size_t r1, const size_t c1, const size_t c2) {
    // A is r1 x c1, B is c1 x c2, C is r1 x c2
    // Implement matrix multiplication logic here
    for (size_t i = 0; i < r1; i++) {
        for (size_t j = 0; j < c2; j++) {
            C[i * c2 + j] = 0;
            for (size_t k = 0; k < c1; k++) {
                C[i * c2 + j] += A[i * c1 + k] * B[k * c2 + j];
            }
        }
    }
}

void mat_elements_product(const double *A, const double *B, double *C, const size_t dim) {
    for (size_t i = 0; i < dim; i++) {
        C[i] = A[i] * B[i];
    }
}

void mat_print(const double *A, const size_t r, const size_t c) {
    // A is r x c
    for (size_t i = 0; i < r; i++) {
        for (size_t j = 0; j < c; j++) {
            printf("%f ", A[i * c + j]);
        }
        printf("\n");
    }
}

void v_add(const double *A, double *B, const size_t dim) {
    // A is r1 x 1, B is r1 x 1, C is r1 x 1
    for (size_t i = 0; i < dim; i++) {
        B[i] += A[i];
    }
}

void v_sub(const double *A, double *B, const size_t dim) {
    // A is r1 x 1, B is r1 x 1, C is r1 x 1
    for (size_t i = 0; i < dim; i++) {
        B[i] -= A[i];
    }
}

void v_clone(const double *A, double *B, const size_t dim) {
    // A is r1 x 1, B is r1 x 1
    for (size_t i = 0; i < dim; i++) {
        B[i] = A[i];
    }
}

void v_apply_function(double *V, const size_t dim, double (*Function)(double)){
    // V is r1 x 1
    for (size_t i = 0; i < dim; i++) {
        V[i] = Function(V[i]);
    }
}

void v_mult_scalar(double *V, const double c, const size_t dim) {
    // V is r1 x 1
    for (size_t i = 0; i < dim; i++) {
        V[i] *= c;
    }
}

// la seguente funzione restituisce 1.0 se x è intero dispari positivo, 0.0 se x è intero pari, -1.0 se x è intero dispari negativo
double isOdd(double x) {
    return (double)((int)x % 2);
}

double sigmoid(double x){
    if(x < -709) return 0; // Protegge da underflow di exp per x molto negativo
    else if(x > 709) return 1; // Protegge da overflow di exp per x molto positivo
    return 1 / (1 + exp(-x));
}

double d_sigmoid(double x){
    return sigmoid(x) * (1 - sigmoid(x));
}

void reset_gradients(NeuralNewtork *nn) {
    memset(nn->d_weights_1, 0, sizeof(double) * INPUTS * NEURONS_1);
    memset(nn->d_bias_1, 0, sizeof(double) * NEURONS_1);
    memset(nn->d_weights_2, 0, sizeof(double) * NEURONS_1 * OUTPUTS);
    memset(nn->d_bias_2, 0, sizeof(double) * OUTPUTS);
}

void update_weights(NeuralNewtork *nn) {
    // Aggiorna i pesi e i bias della rete neurale
    for (size_t i = 0; i < INPUTS * NEURONS_1; i++) {
        nn->weights_1[i] -= LEARNING_RATE * nn->d_weights_1[i];
    }
    for (size_t i = 0; i < NEURONS_1; i++) {
        nn->bias_1[i] -= LEARNING_RATE * nn->d_bias_1[i];
    }
    for (size_t i = 0; i < NEURONS_1 * OUTPUTS; i++) {
        nn->weights_2[i] -= LEARNING_RATE * nn->d_weights_2[i];
    }
    for (size_t i = 0; i < OUTPUTS; i++) {
        nn->bias_2[i] -= LEARNING_RATE * nn->d_bias_2[i];
    }
}

//  La freccia -> in C (e C++) serve per accedere a un membro di una struttura tramite un puntatore.
void feedforward(NeuralNewtork *nn) {
    // Calcola l'input per il primo strato nascosto
    mat_mult(nn->inputs, nn->weights_1, nn->input_neurons_1, 1, INPUTS, NEURONS_1);

    // Aggiungi i bias al risultato
    v_add(nn->bias_1, nn->input_neurons_1, NEURONS_1);

    // Applica sigmoid e salva correttamente in output_neurons_1
    v_clone(nn->input_neurons_1, nn->output_neurons_1, NEURONS_1);
    v_apply_function(nn->output_neurons_1, NEURONS_1, sigmoid);

    // Calcola l'input del neurone di output
    mat_mult(nn->output_neurons_1, nn->weights_2, nn->input_outputs, 1, NEURONS_1, OUTPUTS);

    // Aggiungi bias all'output
    v_add(nn->bias_2, nn->input_outputs, OUTPUTS);

    // Applica sigmoid finale
    v_clone(nn->input_outputs, nn->output_outputs, OUTPUTS);
    v_apply_function(nn->output_outputs, OUTPUTS, sigmoid);
}

void init_NN(NeuralNewtork *nn) {
    // inizializzare i pesi con valori casuali compresi tra -1 e 1.
    for (size_t i = 0; i < INPUTS * NEURONS_1; i++) {
        nn->weights_1[i] = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
    }
    for (size_t i = 0; i < NEURONS_1; i++) {
        nn->bias_1[i] = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
    }
    for (size_t i = 0; i < NEURONS_1 * OUTPUTS; i++) {
        nn->weights_2[i] = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
    }
    for (size_t i = 0; i < OUTPUTS; i++) {
        nn->bias_2[i] = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
    }
}

void backpropagation(NeuralNewtork *nn, double *target){
    // nelle righe successive viene fatta un'operazione di calcolo parallelo per: 
    
    // Copia l'output della rete nel vettore degli errori (per inizializzare il calcolo dell'errore)
    // void v_clone(const double *A, double *B, const size_t dim)
    v_clone(nn->output_outputs, nn->error_outputs, OUTPUTS);

    // Calcola l'errore: errore = output_rete - output_atteso
    //     Dopo questa operazione: error_outputs = output_outputs - expected_outputs
    // void v_sub(const double *A, double *B, const size_t dim)
    v_sub(target, nn->error_outputs, OUTPUTS);
    // occhio all'ordine della sottrazione, v_sub: B = B - A il gradiente avrà il segno invertito

    // Calcola la derivata della funzione di attivazione sigmoid
    //     direttamente nel vettore `input_outputs`
    //     Dopo questa operazione: input_outputs[i] = sigmoid'(input_outputs[i])
    // void v_apply_function(double *V, const size_t dim, double* (*Function)(double))
    v_apply_function(nn->input_outputs, OUTPUTS, d_sigmoid);

    // Calcola l'errore finale moltiplicando:
    //     error_outputs[i] *= d_sigmoid(input_outputs[i])
    // Matematicamente: errore = errore * derivata_sigmoid
    // void mat_elements_product(const double *A, const double *B, double *C, const size_t dim)
    mat_elements_product(nn->error_outputs, nn->input_outputs, nn->error_outputs, OUTPUTS);

    // Calcola il gradiente dei pesi dallo strato nascosto all'output:
    //     dW = hidden_output * errore_output^T
    //     output_neurons_1: (NEURONS_1 x 1)
    //     error_outputs:     (1 x OUTPUTS)
    //     Risultato d_weights_2: (NEURONS_1 x OUTPUTS)
    // void mat_mult(const double *A, const double *B, double *C, const size_t r1, const size_t c1, const size_t c2)
    
    // i pesi dello strato 2 = output dello strato 1 * errore all'uscita*derivata_sigmoid(input dello strato di output))
    mat_mult(nn->output_neurons_1, nn->error_outputs, nn->d_weights_2, NEURONS_1, 1, OUTPUTS);

    // Calcola il gradiente dei bias per lo strato di output:
    //     d_bias += errore_output
    // void v_add(const double *A, double *B, const size_t dim)
    v_add(nn->error_outputs, nn->d_bias_2, OUTPUTS);
    
    //---------------------------------------------------------------------------------------------------

    // Calcola l'errore per lo strato nascosto (strato 1)
    mat_mult(nn->error_outputs, nn->weights_2, nn->error_neurons_1, 1, OUTPUTS, NEURONS_1);
    // Calcola la derivata della funzione di attivazione sigmoid sul vettore `input_neurons_1`
    v_apply_function(nn->input_neurons_1, NEURONS_1, d_sigmoid);
    // Calcola l'errore finale moltiplicando:
    mat_elements_product(nn->error_neurons_1, nn->input_neurons_1, nn->error_neurons_1, NEURONS_1);
    // Calcola il gradiente dei pesi 1
    mat_mult(nn->inputs, nn->error_neurons_1, nn->d_weights_1, INPUTS, 1, NEURONS_1); // corretto

    // Calcola il gradiente dei bias 1
    v_add(nn->error_neurons_1, nn->d_bias_1, NEURONS_1);

    update_weights(nn);

}

double inference(NeuralNewtork *nn, const double *test_data) {
    // 1) Assegna gli input alla rete
    for (size_t i = 0; i < INPUTS; i++) {
        nn->inputs[i] = test_data[i];
    }
    // 2) Calcola feedforward (userà i pesi già appresi)
    feedforward(nn);

    // 3) Restituisci l’uscita del neurone di output
    return nn->output_outputs[0];
}

int main(){

    /* Inizializzazione della memoria: con una rete a 3 input / 1 neurone hidden / 1 output, l'inizializzazione all'interno dell struct va bene così com'è: 
    allocazione automatica, semplice e veloce. Se invece si pensa di voler generalizzare la rete, per reti dinamiche, è consigliato: passare a puntatori (double *) in struct,
    allocare la memoria con malloc e scrivere una init e free per gestire la memoria */
    
    srand((unsigned int)time(NULL)); // Inizializza random

    // 1) Istanzia e inizializza la rete neurale
    NeuralNewtork nn;
    init_NN(&nn);

    // 2) Dataset completo: 8 righe, 3 input + 1 output (XOR a 3 bit)
    double dataset_all[FULL_DATASET_SIZE][4] = {
        {0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 1.0},
        {0.0, 1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0, 0.0},
        {1.0, 0.0, 0.0, 1.0},
        {1.0, 0.0, 1.0, 0.0},
        {1.0, 1.0, 0.0, 0.0},
        {1.0, 1.0, 1.0, 1.0}
    };

    // 3) Array per i 6 campioni di training e i 2 di test
    double train_data[TRAIN_DATASET_SIZE][4];
    double test_data[TEST_DATASET_SIZE][4];

    // 4) Crea un array di indici [0..7] e fai shuffle
    int indices[FULL_DATASET_SIZE];
    for(int i = 0; i < FULL_DATASET_SIZE; i++){
        indices[i] = i;
    }

    // Fisher-Yates shuffle
    for(int i = FULL_DATASET_SIZE - 1; i > 0; i--){
        int j = rand() % (i + 1);
        // scambia indices[i] e indices[j]
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    // 5) Copia i primi 6 (shuffle) in train_data, i rimanenti 2 in test_data
    for(int i = 0; i < TRAIN_DATASET_SIZE; i++){
        int idx = indices[i];
        for(int j = 0; j < INPUTS+1; j++){
            train_data[i][j] = dataset_all[idx][j];
        }
    }
    for(int i = 0; i < TEST_DATASET_SIZE; i++){
        int idx = indices[TRAIN_DATASET_SIZE + i];
        for(int j = 0; j < INPUTS+1; j++){
            test_data[i][j] = dataset_all[idx][j];
        }
    }

    // Mostriamo i pattern scelti
    printf("TRAINING SET (%d pattern):\n", TRAIN_DATASET_SIZE);
    for(int i = 0; i < TRAIN_DATASET_SIZE; i++){
        printf("%.0f %.0f %.0f -> %.0f\n",
               train_data[i][0], train_data[i][1],
               train_data[i][2], train_data[i][3]);
    }
    printf("\nTEST SET (%d pattern):\n", TEST_DATASET_SIZE);
    for(int i = 0; i < TEST_DATASET_SIZE; i++){
        printf("%.0f %.0f %.0f -> %.0f\n",
               test_data[i][0], test_data[i][1],
               test_data[i][2], test_data[i][3]);
    }
    printf("\n");

    // Addestramento
    printf("Avvio Addestramento:\n");
    for(int epoch = 0; epoch < EPOCHS; epoch++){
        double total_error = 0.0;
        
        // Imposta gli input del network
        for(int i = 0; i < TRAIN_DATASET_SIZE; i++){
            for(int j = 0; j < INPUTS; j++){
                nn.inputs[j] = train_data[i][j];
            }
        
            feedforward(&nn);
            
            // printf("Output del neurone di output: %.4f\n", nn.output_outputs[0]);

            // Calcolo dell'errore e backpropagation
            double target[1];
            target[0] = train_data[i][3];
            backpropagation(&nn, target);

            // Accumula errore (semplice somma di errori di output)
            // (avendo 1 solo neurone in uscita, ci basta guardare error_outputs[0])
            for (int i = 0; i < OUTPUTS; i++){
                total_error += fabs(nn.error_outputs[i]);
            }
        }
        
        reset_gradients(&nn); // Aggiunta importante per stabilità!
        // Errore medio
        total_error /= (double)TRAIN_DATASET_SIZE;

        // Stampa
        printf("Epoch %d, Errore medio: %.4f\n", epoch, total_error);

        // Stop se l'errore è sotto la soglia
        if(total_error < ERROR_THRESHOLD){
            printf("Soglia di errore raggiunta. ");
            break;
        }
    }
    printf("Fine Addestramento:\n");

    // Inferenza sul modello addestrato
    printf("Calcolo inferenza:\n");
    // esempio output: 0 1 1 -> 0.9524 (atteso: 0)
    for (size_t i = 0; i < TEST_DATASET_SIZE; i++){
        double risultato = inference(&nn, test_data[i]);
        printf("%.0f %.0f %.0f -> %.4f (atteso: %.0f)\n",
               test_data[i][0], test_data[i][1],
               test_data[i][2], risultato, test_data[i][3]);
    }

    return 0;
}
